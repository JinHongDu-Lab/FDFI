"""Focused inexpensive tests for replication orchestration and pure helpers."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_quick_paths_cannot_overwrite_formal_paths():
    from replication.scripts.common import RunConfig, FIGURES_DIR
    with tempfile.TemporaryDirectory() as raw:
        config = RunConfig.create("quick", run_id="quick-test", runs_dir=Path(raw))
        assert config.figures_dir != FIGURES_DIR
        assert config.output_root.name == "quick-test"
        assert config.figures_dir.is_relative_to(config.output_root)


def test_incorrect_ctg_checksum_is_blocked():
    from replication.scripts.common import validate_ctg_input
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "CTG.xls"; path.write_bytes(b"not the CTG spreadsheet")
        ok, detail = validate_ctg_input(path)
        assert not ok
        assert "checksum mismatch" in detail


def test_fixed_ctg_workbook_checksum_schema_features_and_classes():
    from replication.scripts.common import CTG_SHA256, DATA_DIR, sha256_file, validate_ctg_input
    from replication.scripts.reproduce_ctg import FEATURES, clean_ctg_frame
    path = DATA_DIR / "CTG.xls"
    assert sha256_file(path) == CTG_SHA256
    assert f"{CTG_SHA256}  CTG.xls" in (DATA_DIR / "checksums.sha256").read_text().splitlines()
    book = pd.ExcelFile(path, engine="xlrd")
    assert "Data" in book.sheet_names
    raw = pd.read_excel(path, sheet_name="Data", skiprows=1, header=0, engine="xlrd")
    assert raw.shape == (2129, 46)
    X, y = clean_ctg_frame(raw)
    assert X.shape == (2126, 21) and list(X.columns) == FEATURES
    assert pd.Series(y).value_counts().sort_index().to_dict() == {0: 1655, 1: 471}
    ok, detail = validate_ctg_input(path)
    assert ok, detail


def test_fixed_sens50_checksums_dimensions_and_group_coverage():
    from replication.scripts.common import (CASE_STUDY_DATA_DIR, DATA_DIR,
                                            FEATURE_GROUP_SHA256, SENS50_SHA256,
                                            sha256_file, validate_sens50_inputs)
    data = CASE_STUDY_DATA_DIR / "sens50_processed_dataset.csv"
    groups = CASE_STUDY_DATA_DIR / "feature_group.csv"
    assert sha256_file(data) == SENS50_SHA256
    assert sha256_file(groups) == FEATURE_GROUP_SHA256
    checksum_lines = (DATA_DIR / "checksums.sha256").read_text().splitlines()
    assert f"{SENS50_SHA256}  ../../docs/case_studies/data/sens50_processed_dataset.csv" in checksum_lines
    assert f"{FEATURE_GROUP_SHA256}  ../../docs/case_studies/data/feature_group.csv" in checksum_lines
    ok, detail = validate_sens50_inputs(data, groups)
    assert ok, detail


def test_ctg_cleaning_produces_2126_by_21_and_binary_outcome():
    from replication.scripts.reproduce_ctg import clean_ctg_frame
    columns = [f"c{i}" for i in range(46)]; columns[-1] = "NSP"
    raw = pd.DataFrame(np.ones((2126, 46)), columns=columns)
    raw.loc[1, "NSP"] = 2; raw.loc[2, "NSP"] = 3
    X, y = clean_ctg_frame(raw)
    assert X.shape == (2126, 21)
    assert set(np.unique(y)) == {0, 1}


def test_sens50_formal_paths_use_fixed_case_study_data():
    from replication.scripts import reproduce_sens50_eot as sens
    assert sens.DATA_PATH == ROOT / "docs/case_studies/data/sens50_processed_dataset.csv"
    assert sens.GROUP_PATH == ROOT / "docs/case_studies/data/feature_group.csv"


def test_simulation_allows_only_draft_quick_run_before_author_approval():
    from replication.scripts import reproduce_simulation
    quick = reproduce_simulation.preflight("quick")
    full = reproduce_simulation.preflight("full")
    assert quick["ready"] is True
    assert "meeting-preview" in str(quick["detail"])
    assert full["ready"] is False
    assert "BLOCKED" in str(full["detail"])


def test_simulation_block_covariance_and_relevant_features_are_deterministic():
    from replication.scripts.simulation import block_covariance, generate_data
    covariance = block_covariance(10, 5, 0.6)
    assert np.allclose(np.diag(covariance), 1.0)
    assert covariance[0, 1] == 0.6 and covariance[0, 5] == 0.0
    first = generate_data(40, 0.6, 7, 10, 5, (0, 1, 5, 6), (1.5, 1.0, 1.5, 1.0), 1.0)
    second = generate_data(40, 0.6, 7, 10, 5, (0, 1, 5, 6), (1.5, 1.0, 1.5, 1.0), 1.0)
    assert np.array_equal(first[0], second[0])
    assert first[2].sum() == 4


def test_simulation_benchmark_contract_uses_both_fdfi_resampling_versions():
    from replication.scripts.simulation import (
        BASELINE_METHODS, FDFI_FAMILIES, RESAMPLING_VERSIONS, settings,
    )
    from replication.scripts.simulation_plot import METHODS
    assert BASELINE_METHODS == ("LOCO", "CPI")
    assert FDFI_FAMILIES == ("DFI-OT", "DFI-EOT", "FDFI")
    assert RESAMPLING_VERSIONS == ("cpi", "scpi")
    assert METHODS == ("LOCO", "CPI", "DFI-OT", "DFI-EOT", "FDFI")
    quick = settings("quick")
    assert len(quick["n_values"]) == 5
    assert len(quick["rho_values"]) == 4


def test_full_strict_preflight_stops_before_workflows_or_outputs():
    from replication.scripts.common import PreflightReport, run_replication
    called = []
    module_name = "tests.fake_never_runs"
    sys.modules[module_name] = SimpleNamespace(run=lambda **kwargs: called.append(True))
    try:
        with tempfile.TemporaryDirectory() as raw:
            runs = Path(raw) / "runs"
            code = run_replication("full", workflow_defs=(("x", "fake", module_name),),
                                   preflight_fn=lambda *args: PreflightReport(blockers=["blocked"]),
                                   runs_dir=runs)
            assert code == 2
            assert called == []
            assert not runs.exists()
    finally:
        sys.modules.pop(module_name, None)


def test_selected_full_workflow_is_also_stopped_by_preflight_blockers():
    from replication.scripts.common import PreflightReport, run_replication
    called = []
    module_name = "tests.fake_selected_never_runs"
    sys.modules[module_name] = SimpleNamespace(run=lambda **kwargs: called.append(True))
    try:
        with tempfile.TemporaryDirectory() as raw:
            runs = Path(raw) / "runs"
            code = run_replication(
                "full",
                workflow="sens50",
                workflow_defs=(("sens50", "fake", module_name),),
                preflight_fn=lambda *args: PreflightReport(blockers=["version mismatch"]),
                runs_dir=runs,
            )
            assert code == 2
            assert called == []
            assert not runs.exists()
    finally:
        sys.modules.pop(module_name, None)


def test_workflow_status_and_aggregate_status_are_preserved():
    from replication.scripts.common import WorkflowResult, aggregate_status
    assert aggregate_status([WorkflowResult("a", "FAILED")]) == "FAILED"
    assert aggregate_status([WorkflowResult("a", "BLOCKED")]) == "BLOCKED"
    assert aggregate_status([WorkflowResult("a", "SUCCESS")]) == "SUCCESS"
    assert aggregate_status([WorkflowResult("a", "SUCCESS"), WorkflowResult("b", "BLOCKED")]) == "BLOCKED"


def test_returned_blocked_status_stays_blocked_in_manifest():
    from replication.scripts.common import PreflightReport, WorkflowResult, run_replication
    module_name = "tests.fake_returned_blocked"
    sys.modules[module_name] = SimpleNamespace(
        run=lambda mode, config: WorkflowResult("blocked stage", "BLOCKED", error="not ready")
    )
    try:
        with tempfile.TemporaryDirectory() as raw:
            runs = Path(raw) / "runs"
            code = run_replication("quick", workflow_defs=(("x", "blocked stage", module_name),),
                                   preflight_fn=lambda *args: PreflightReport(), runs_dir=runs)
            assert code == 2
            run_dir = next(runs.iterdir())
            manifest = json.loads((run_dir / "metadata/run_manifest.json").read_text())
            assert manifest["status"] == "BLOCKED"
            assert manifest["workflows"][0]["status"] == "BLOCKED"
    finally:
        sys.modules.pop(module_name, None)


def test_success_log_and_manifest_exclude_stale_artifacts():
    from replication.scripts.common import PreflightReport, WorkflowResult, run_replication
    module_name = "tests.fake_success_workflow"
    def fake_run(mode, config):
        path = config.tables_dir / "current.csv"; path.write_text("x\n1\n", encoding="utf-8")
        return WorkflowResult("fake stage", "SUCCESS", generated_files=[str(path)])
    sys.modules[module_name] = SimpleNamespace(run=fake_run)
    try:
        with tempfile.TemporaryDirectory() as raw:
            runs = Path(raw) / "runs"
            stale = runs / "old-run/figures/stale.pdf"
            stale.parent.mkdir(parents=True); stale.write_bytes(b"stale")
            code = run_replication("quick", workflow_defs=(("fake", "fake stage", module_name),),
                                   preflight_fn=lambda *args: PreflightReport(), runs_dir=runs)
            assert code == 0
            run_dirs = [path for path in runs.iterdir() if path.name != "old-run"]
            assert len(run_dirs) == 1
            manifest = json.loads((run_dirs[0] / "metadata/run_manifest.json").read_text())
            paths = [item["path"] for item in manifest["artifacts"]]
            assert not any("stale.pdf" in path for path in paths)
            assert any("current.csv" in path for path in paths)
            assert manifest["status"] == "SUCCESS"
            log = next((run_dirs[0] / "logs").glob("*.log")).read_text()
            assert "--- fake stage ---" in log
            assert "Overall: SUCCESS" in log
            assert len(log) > 100
            from replication.scripts.common import sha256_file
            log_entry = next(item for item in manifest["artifacts"] if item["kind"] == "log")
            assert log_entry["sha256"] == sha256_file(next((run_dirs[0] / "logs").glob("*.log")))
            assert manifest["workflow_selection"] == "all"
            assert manifest["formal_manuscript_run"] is False
            assert manifest["publication_attempted"] is False
            assert manifest["provenance"]["source_notebooks"]
            assert manifest["provenance"]["workflow_scripts"]
            inputs = {Path(item["path"]).name: item for item in manifest["provenance"]["input_data"]}
            assert inputs["sens50_processed_dataset.csv"]["sha256"]
            assert inputs["feature_group.csv"]["sha256"]
            assert manifest["workflows"][0]["process_peak_memory_mb"] is not None
    finally:
        sys.modules.pop(module_name, None)


def test_ctg_notebook_equivalent_configuration_and_colours():
    from replication.scripts import reproduce_ctg as ctg
    assert ctg.DATA_MODEL_SEED == 42
    assert ctg.FLOW_SEED == 0
    assert ctg.LOSS == "squared_error"
    assert ctg.METHOD == "cpi"
    assert ctg.FULL_NSAMPLES == 50
    assert ctg.FULL_FLOW_STEPS == 5000
    assert ctg.NOTEBOOK_EXPECTED["feature_rejections"] == 18
    assert ctg.NOTEBOOK_EXPECTED["group_rejections"] == 1
    assert set(ctg.FEATURES) == set(ctg.GROUP_COLOUR)
    for key in ("LB", "ASTV", "AC", "Width", "Mode", "Median"):
        assert key in ctg.GROUP_COLOUR


def test_ctg_inference_and_sampling_settings_share_explicit_configuration():
    from replication.scripts import reproduce_ctg as ctg
    import inspect
    required = {"alpha", "target", "alternative", "multitest_method", "threshold_null",
                "var_floor_c", "var_floor_method", "var_floor_quantile", "margin",
                "margin_method", "margin_quantile", "verbose"}
    assert set(ctg.FEATURE_INFERENCE) == required
    assert set(ctg.GROUP_INFERENCE) == required
    settings = ctg._intended_settings("quick")["intended"]
    assert settings["feature_inference"] == ctg.FEATURE_INFERENCE
    assert settings["group_inference"] == ctg.GROUP_INFERENCE
    assert settings["flow_explainer"]["loss"] == ctg.LOSS
    assert settings["flow_explainer"]["method"] == ctg.METHOD
    assert settings["flow_explainer"]["configuration_source"] == (
        "verified FDFI 0.0.10 notebook baseline"
    )
    source = inspect.getsource(ctg.run)
    assert "conf_int(**FEATURE_INFERENCE)" in source
    assert "conf_int(groups=group_indices, **GROUP_INFERENCE)" in source
    assert "loss=LOSS, method=METHOD" in source
    assert 'flow_solver_method=FLOW_SOLVER["method"]' in source
    assert 'diagnostics_solver_method=DIAGNOSTICS_SOLVER["method"]' in source
    assert "results = explainer(X)" in source
    assert "explain_batches" not in source


def test_failed_ctg_attempt_retains_intended_settings():
    from replication.scripts import reproduce_ctg as ctg
    from replication.scripts.common import RunConfig, WorkflowBlocked
    original = ctg._load
    try:
        ctg._load = lambda: (_ for _ in ()).throw(WorkflowBlocked("synthetic failure"))
        with tempfile.TemporaryDirectory() as raw:
            config = RunConfig.create("quick", run_id="ctg-failure", runs_dir=Path(raw)); config.ensure_directories()
            try:
                ctg.run("quick", config)
            except WorkflowBlocked:
                pass
            payload = json.loads((config.metadata_dir / "ctg_settings.json").read_text())
            assert payload["configuration_status"] == "INTENDED"
            assert payload["observed"] is None
            assert payload["intended"]["smoke_test_only"] is True
    finally:
        ctg._load = original


def test_failed_sens50_attempt_retains_intended_settings():
    from replication.scripts import reproduce_sens50_eot as sens
    from replication.scripts.common import RunConfig, WorkflowBlocked
    original = sens.DATA_PATH
    try:
        with tempfile.TemporaryDirectory() as raw:
            sens.DATA_PATH = Path(raw) / "missing.csv"
            config = RunConfig.create("quick", run_id="sens-failure", runs_dir=Path(raw) / "runs")
            config.ensure_directories()
            try:
                sens.run("quick", config)
            except WorkflowBlocked:
                pass
            payload = json.loads((config.metadata_dir / "sens50_settings.json").read_text())
            assert payload["configuration_status"] == "INTENDED"
            assert payload["observed"] is None
    finally:
        sens.DATA_PATH = original


def test_ctg_optional_axis_matches_notebook_logic():
    from replication.scripts.reproduce_ctg import optional_summary_axis
    positive = optional_summary_axis(np.array([0.0, .1, .2, 1.0]))
    mixed = optional_summary_axis(np.array([-.2, 0.0, .1, .4]))
    assert positive["scale"] == "log" and positive["xmin"] > 0
    assert mixed["scale"] == "symlog" and mixed["linthresh"] > 0


def test_sens50_quick_subset_covers_every_group():
    from replication.scripts.reproduce_sens50_eot import select_quick_features
    names = [f"f{i}" for i in range(40)]
    X = pd.DataFrame(np.zeros((3, 40)), columns=names)
    groups = pd.DataFrame(False, index=names, columns=[f"g{i}" for i in range(14)])
    for i in range(14): groups.iloc[i, i] = True
    selected = select_quick_features(X, groups, budget=20)
    assert len(selected) == 20
    assert groups.loc[selected].any(axis=0).all()


def test_sens50_explicit_estimand_settings_and_returned_ranking_are_preserved():
    from replication.scripts import reproduce_sens50_eot as sens
    import inspect
    assert sens.LOSS == "squared_error"
    assert sens.METHOD == "cpi"
    assert sens.FEATURE_INFERENCE == {
        "alpha": .05,
        "var_floor_c": .1,
        "var_floor_method": "mixture",
        "var_floor_quantile": .95,
        "margin": 0.0,
        "margin_method": "auto",
        "margin_quantile": .95,
        "alternative": "two-sided",
        "verbose": False,
    }
    assert sens.GROUP_INFERENCE == {
        "alpha": .05,
        "threshold_null": True,
        "var_floor_c": .1,
        "var_floor_method": "fixed",
        "margin": 0.0,
        "margin_method": "fixed",
        "alternative": "two-sided",
        "multitest_method": "bonferroni",
        "verbose": False,
    }
    assert sens.NOTEBOOK_EXPECTED == {
        "phi_x_sum": 0.7396973904611562,
        "phi_z_sum": 0.7396975752007999,
        "x_feature_rejections": 6,
        "z_feature_rejections": 7,
        "x_group_rejections": 10,
        "z_group_rejections": 11,
    }
    result = {
        "score": np.array([10.0, 2.0]),
        "se": np.array([10.0, .2]),
        "zscore": np.array([1.0, 10.0]),
        "ranking": np.array([2, 1]),
        "ci_lower": np.array([-1.0, 1.0]),
        "ci_upper": np.array([21.0, 3.0]),
        "pvalue": np.array([.3, 0.0]),
        "pvalue_adj": np.array([.3, 0.0]),
        "reject_null": np.array([False, True]),
    }
    frame = sens._frame(result, ["raw-score-first", "z-score-first"], "group")
    assert frame["ranking"].tolist() == [2, 1]
    source = inspect.getsource(sens.run)
    assert "loss=LOSS, method=METHOD" in source
    assert "conf_int(target=target, **FEATURE_INFERENCE)" in source
    assert "conf_int(target=target, groups=group_dict, **GROUP_INFERENCE)" in source
    plot_source = inspect.getsource(sens._plot_groups)
    assert "axvline" in plot_source
    assert 'label="p=0.05"' in plot_source


def test_sens50_does_not_export_notebook_absent_diagnostics():
    import inspect
    from replication.scripts import reproduce_sens50_eot as sens
    source = inspect.getsource(sens)
    assert "sens50_eot_diagnostics" not in source
    assert "sens50_eot_top_latent_dependencies" not in source
    assert "sens50_eot_latent_dcor_matrix" not in source


def test_sens50_writes_public_summary_output_for_human_reading():
    from replication.scripts.common import RunConfig
    from replication.scripts.reproduce_sens50_eot import (
        FEATURE_INFERENCE,
        GROUP_INFERENCE,
        _write_notebook_summary_tables,
    )

    calls = []

    class FakeExplainer:
        def summary(self, **kwargs):
            calls.append(kwargs)
            level = "group" if "groups" in kwargs else "feature"
            return f"package summary: {kwargs['target']} {level}"

    with tempfile.TemporaryDirectory() as raw:
        config = RunConfig.create(
            "quick", run_id="summary-test", runs_dir=Path(raw)
        )
        config.ensure_directories()
        artifacts = _write_notebook_summary_tables(
            FakeExplainer(), {"g": [0, 1]}, config
        )
        feature_text = (
            config.tables_dir / "sens50_eot_feature_summary.txt"
        ).read_text()
        group_text = (
            config.tables_dir / "sens50_eot_group_summary.txt"
        ).read_text()

    assert len(artifacts) == 2
    assert feature_text == (
        "=== X-space feature summary ===\npackage summary: X feature\n\n"
        "=== Z-space feature summary ===\npackage summary: Z feature\n"
    )
    assert group_text == (
        "=== X-space group summary ===\npackage summary: X group\n\n"
        "=== Z-space group summary ===\npackage summary: Z group\n"
    )
    feature_calls = [call for call in calls if "groups" not in call]
    group_calls = [call for call in calls if "groups" in call]
    assert all(call["print_output"] is False for call in calls)
    assert all(
        all(call[key] == value for key, value in FEATURE_INFERENCE.items())
        for call in feature_calls
    )
    assert all(
        all(call[key] == value for key, value in GROUP_INFERENCE.items())
        for call in group_calls
    )


def test_transactional_publication_publishes_key_results_and_removes_only_managed_stale():
    from replication.scripts.common import RunConfig, _publish_formal
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw); config = RunConfig.create("full", run_id="publish-test", runs_dir=root / "runs")
        config.ensure_directories()
        (config.tables_dir / "key_numerical_results.csv").write_text("metric,value\na,1\n")
        (config.figures_dir / "new.pdf").write_bytes(b"new")
        figures, tables, results = root / "stable/figures", root / "stable/tables", root / "stable/results"
        for path in (figures, tables, results): path.mkdir(parents=True)
        stale = figures / "stale-owned.pdf"; stale.write_bytes(b"stale")
        reference = figures / "reference.pdf"; reference.write_bytes(b"reference")
        state = root / "stable/formal_publication.json"
        state.write_text(json.dumps({"managed_paths": [str(stale)]}))
        published = _publish_formal(config, figures, tables, results, state)
        assert not stale.exists()
        assert reference.read_bytes() == b"reference"
        assert (figures / "new.pdf").is_file()
        assert (tables / "key_numerical_results.csv").is_file()
        assert any("key_numerical_results.csv" in path for path in published)


def test_formal_manifest_flags_require_full_all_success_and_publication():
    from replication.scripts import common
    from replication.scripts.common import PreflightReport, WorkflowResult, run_replication
    module_name = "tests.fake_formal_success"
    def fake_run(mode, config):
        path = config.tables_dir / "result.csv"; path.write_text("x\n1\n")
        settings = config.metadata_dir / "fake_settings.json"; settings.write_text('{"seed": 1}\n')
        return WorkflowResult("fake", "SUCCESS", [str(path), str(settings)])
    sys.modules[module_name] = SimpleNamespace(run=fake_run)
    original_git = common._git
    common._git = lambda *args: " M synthetic-dirty-file" if args == ("status", "--porcelain") else original_git(*args)
    try:
        with tempfile.TemporaryDirectory() as raw:
            runs = Path(raw) / "runs"; seen = {}
            def publish(config):
                seen["key_exists_before_publish"] = (config.tables_dir / "key_numerical_results.csv").is_file()
                return ["replication/tables/key_numerical_results.csv"]
            code = run_replication("full", workflow="all",
                                   workflow_defs=(("fake", "fake", module_name),),
                                   preflight_fn=lambda *args: PreflightReport(), publish_fn=publish,
                                   runs_dir=runs)
            manifest = json.loads((next(runs.iterdir()) / "metadata/run_manifest.json").read_text())
            assert code == 0 and seen["key_exists_before_publish"]
            assert manifest["formal_manuscript_run"] is False
            assert manifest["publication_attempted"] is True
            assert manifest["publication_succeeded"] is True
            assert manifest["published_stable_paths"]
            assert "fake_settings.json" in manifest["provenance"]["settings"]
            assert manifest["provenance"]["source_notebooks"]
            execution = {Path(item["path"]).name for item in manifest["provenance"]["execution_files"]}
            assert {"reproduce.py", "reproduce_quick.py", "common.py", "fetch_ctg.py",
                    "environment.yml", "checksums.sha256"}.issubset(execution)
            assert manifest["provenance"]["git"]["dirty"] is True
    finally:
        common._git = original_git
        sys.modules.pop(module_name, None)


def test_selected_workflow_is_never_formal_run():
    from replication.scripts.common import PreflightReport, WorkflowResult, run_replication
    module_name = "tests.fake_selected_success"
    sys.modules[module_name] = SimpleNamespace(run=lambda mode, config: WorkflowResult("fake", "SUCCESS"))
    try:
        with tempfile.TemporaryDirectory() as raw:
            runs = Path(raw) / "runs"
            code = run_replication("full", workflow="sens50",
                                   workflow_defs=(("fake", "fake", module_name),),
                                   preflight_fn=lambda *args: PreflightReport(), runs_dir=runs)
            manifest = json.loads((next(runs.iterdir()) / "metadata/run_manifest.json").read_text())
            assert code == 0
            assert manifest["formal_manuscript_run"] is False
            assert manifest["publication_attempted"] is False
    finally:
        sys.modules.pop(module_name, None)


def test_thread_configuration_and_stage_memory_schema():
    from replication.scripts.common import THREAD_ENV_VARS
    for entry in (ROOT / "replication/reproduce.py", ROOT / "replication/reproduce_quick.py"):
        text = entry.read_text()
        assert 'os.environ[_name] = "1"' in text
    assert len(THREAD_ENV_VARS) == 5
    assert "stage_peak_cpu_rss_mb_sampled" in (ROOT / "replication/scripts/common.py").read_text()
