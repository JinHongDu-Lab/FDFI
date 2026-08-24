"""Shared configuration, strict preflight, logging, and run orchestration."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, TextIO

Mode = Literal["full", "quick"]
Status = Literal["SUCCESS", "FAILED", "BLOCKED", "SKIPPED"]
WorkflowSelection = Literal["all", "sens50", "ctg", "simulation"]

REPLICATION_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REPLICATION_DIR.parent
DATA_DIR = REPLICATION_DIR / "data"
CASE_STUDY_DIR = REPO_ROOT / "docs" / "case_studies"
CASE_STUDY_DATA_DIR = CASE_STUDY_DIR / "data"
EOT_NOTEBOOK = CASE_STUDY_DIR / "eot_case_study_sens50.ipynb"
CTG_NOTEBOOK = CASE_STUDY_DIR / "flow_case_study_ctg.ipynb"
RUNS_DIR = REPLICATION_DIR / "runs"
FIGURES_DIR = REPLICATION_DIR / "figures"
TABLES_DIR = REPLICATION_DIR / "tables"
RESULTS_DIR = REPLICATION_DIR / "results"
CTG_SHA256 = "d6aa62e82625e59f9d5b05bc8fc52af9ea67bdf2a23f1faca8511d5618fcb421"
SENS50_SHA256 = "a7831f7926706377ed9b81fd6c73f995f69f1d971374859828c9300b566a0287"
FEATURE_GROUP_SHA256 = "fc519731d770ca4672029877fbcb884a83f859e687851d0390908c15bbc80e97"
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

WORKFLOWS: tuple[tuple[str, str, str], ...] = (
    ("sens50", "sens50 EOT case study", "scripts.reproduce_sens50_eot"),
    ("ctg", "CTG Flow case study", "scripts.reproduce_ctg"),
    ("simulation", "manuscript simulation study", "scripts.reproduce_simulation"),
)


@dataclass(frozen=True)
class RunConfig:
    mode: Mode
    seed: int
    run_id: str
    output_root: Path
    figures_dir: Path
    tables_dir: Path
    results_dir: Path
    logs_dir: Path
    metadata_dir: Path

    @classmethod
    def create(cls, mode: Mode, run_id: str | None = None, runs_dir: Path = RUNS_DIR) -> "RunConfig":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        identifier = run_id or f"{mode}-{stamp}"
        root = runs_dir / identifier
        return cls(mode, 0 if mode == "full" else 20260721, identifier, root,
                   root / "figures", root / "tables", root / "results",
                   root / "logs", root / "metadata")

    def ensure_directories(self) -> None:
        for path in (self.figures_dir, self.tables_dir, self.results_dir,
                     self.logs_dir, self.metadata_dir):
            path.mkdir(parents=True, exist_ok=False)


@dataclass
class WorkflowResult:
    name: str
    status: Status
    generated_files: list[str] = field(default_factory=list)
    key_results: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    seeds: dict[str, int] = field(default_factory=dict)
    started_at: str = ""
    ended_at: str = ""
    runtime_seconds: float = 0.0
    process_peak_memory_mb: float | None = None
    cuda_peak_allocated_mb: float | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "SUCCESS"


@dataclass
class PreflightReport:
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockers


class WorkflowBlocked(RuntimeError):
    """Raised when a documented requirement is unavailable."""


class _Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (ImportError, AttributeError, RuntimeError):
        pass


class PeakRSSMonitor:
    """Sample this process's RSS during one workflow stage."""
    def __init__(self, interval_seconds: float = .01):
        self.interval_seconds = interval_seconds; self.peak_bytes: int | None = None
        self._stop = threading.Event(); self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            return
        def sample() -> None:
            while not self._stop.is_set():
                try:
                    rss = process.memory_info().rss
                    self.peak_bytes = rss if self.peak_bytes is None else max(self.peak_bytes, rss)
                except (OSError, psutil.Error):
                    pass
                self._stop.wait(self.interval_seconds)
        self._thread = threading.Thread(target=sample, name="replication-rss-monitor", daemon=True)
        self._thread.start()

    def stop_mb(self) -> float | None:
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout=1)
        return None if self.peak_bytes is None else self.peak_bytes / (1024 ** 2)


def cuda_peak_allocated_mb(reset: bool = False) -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return 0.0
        return float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
    except (ImportError, RuntimeError):
        return None


def validate_ctg_input(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing {relative(path)}; run python replication/scripts/fetch_ctg.py"
    actual = sha256_file(path)
    if actual != CTG_SHA256:
        return False, f"checksum mismatch for {relative(path)}: expected {CTG_SHA256}, got {actual}"
    try:
        import pandas as pd
        book = pd.ExcelFile(path, engine="xlrd")
        if "Data" not in book.sheet_names:
            return False, f"Data sheet missing from {relative(path)}; sheets={book.sheet_names}"
        raw = pd.read_excel(path, sheet_name="Data", skiprows=1, header=0, engine="xlrd")
        expected_features = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV",
                             "ALTV", "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros",
                             "Mode", "Mean", "Median", "Variance", "Tendency"]
        if raw.shape != (2129, 46) or "NSP" not in raw.columns:
            return False, f"invalid raw CTG schema: shape={raw.shape}, NSP={'NSP' in raw.columns}"
        frame = pd.concat([raw.iloc[:, 10:31].copy(), raw["NSP"].copy()], axis=1)
        # Pandas disambiguates repeated workbook headers as AC.1, FM.1, etc.;
        # the notebook assigns the documented analysis names positionally.
        frame.columns = expected_features + ["NSP"]
        frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
        y = (frame["NSP"].astype(int) != 1).astype(int)
        counts = y.value_counts().sort_index().to_dict()
        if frame.shape != (2126, 22) or counts != {0: 1655, 1: 471}:
            return False, f"invalid cleaned CTG data: X_shape={(frame.shape[0], 21)}, classes={counts}"
    except Exception as exc:
        return False, f"cannot read/validate CTG workbook: {exc}"
    return True, (f"{relative(path)} SHA-256={actual}; Data raw=2129x46, "
                  "cleaned=2126x21, classes Normal=1655/Other=471")


def validate_sens50_inputs(data_path: Path, group_path: Path) -> tuple[bool, str]:
    for path, expected in ((data_path, SENS50_SHA256), (group_path, FEATURE_GROUP_SHA256)):
        if not path.is_file(): return False, f"missing {relative(path)}"
        actual = sha256_file(path)
        if actual != expected: return False, f"checksum mismatch for {relative(path)}: expected {expected}, got {actual}"
    try:
        import pandas as pd
        data = pd.read_csv(data_path); groups = pd.read_csv(group_path)
        if data.shape != (611, 833) or "sens50" not in data.columns:
            return False, f"invalid sens50 schema: shape={data.shape}, outcome={'sens50' in data.columns}"
        predictors = set(data.columns) - {"sens50"}
        if not {"feature", "group"}.issubset(groups.columns): return False, "feature_group.csv lacks feature/group columns"
        missing = predictors - set(groups["feature"])
        if missing or groups["group"].nunique() != 14:
            return False, f"invalid group coverage: missing={len(missing)}, groups={groups['group'].nunique()}"
    except Exception as exc:
        return False, f"cannot validate sens50 inputs: {exc}"
    return True, (f"fixed copies valid: data={SENS50_SHA256}, groups={FEATURE_GROUP_SHA256}; "
                  "611 observations, 832 predictors, 14 groups")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def _pinned_versions() -> dict[str, str]:
    aliases = {"python": "python", "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
               "scikit-learn": "scikit-learn", "matplotlib": "matplotlib", "seaborn": "seaborn",
               "statsmodels": "statsmodels", "openpyxl": "openpyxl", "xlrd": "xlrd",
               "psutil": "psutil", "tqdm": "tqdm", "pip": "pip",
               "adjusttext": "adjustText", "ipykernel": "ipykernel",
               "nbformat": "nbformat", "nbclient": "nbclient", "nbconvert": "nbconvert",
               "pytest": "pytest", "torch": "torch", "torchdiffeq": "torchdiffeq"}
    pins: dict[str, str] = {}
    for raw in (REPLICATION_DIR / "environment.yml").read_text(encoding="utf-8").splitlines():
        text = raw.strip().lstrip("-").strip()
        if "==" in text:
            name, version = text.split("==", 1)
        elif "=" in text and not text.startswith(("name:", "channels:")):
            name, version = text.split("=", 1)
        else:
            continue
        name = name.strip()
        if name in aliases:
            pins[aliases[name]] = version.strip()
    return pins


def _selected(selection: WorkflowSelection) -> tuple[tuple[str, str, str], ...]:
    return WORKFLOWS if selection == "all" else tuple(item for item in WORKFLOWS if item[0] == selection)


def preflight_check(mode: Mode = "full", workflow: WorkflowSelection = "all",
                    *, ctg_path: Path | None = None, emit: bool = True,
                    runs_dir: Path = RUNS_DIR) -> PreflightReport:
    """Validate readiness without training models or computing attribution."""
    report = PreflightReport()
    selected = {item[0] for item in _selected(workflow)}

    def check(ok: bool, label: str, detail: str, *, warning: bool = False) -> None:
        state = "OK" if ok else ("WARN" if warning else "BLOCKED")
        if emit:
            print(f"[{state}] {label}: {detail}")
        if not ok:
            (report.warnings if warning else report.blockers).append(f"{label}: {detail}")

    if emit:
        print("FDFI JSS replication preflight")
        print(f"Mode: {mode}; workflow: {workflow}")
        print(f"Git commit: {_git('rev-parse', 'HEAD')}")
        print(f"Git worktree: {'dirty' if _git('status', '--porcelain') else 'clean'}")

    dirty_files = [line for line in _git("status", "--porcelain").splitlines() if line]
    if dirty_files:
        check(False, "Git clean-worktree requirement",
              f"{len(dirty_files)} dirty/untracked entries; final full/all runs require a clean tree",
              warning=mode == "quick" or workflow != "all")

    pins = _pinned_versions()
    for name, expected in pins.items():
        actual = platform.python_version() if name == "python" else _package_version(name)
        present = actual != "MISSING"
        check(present, f"dependency {name}", actual)
        matches_pin = actual == expected or (name == "python" and actual.startswith(expected + "."))
        if present and not matches_pin:
            # Every full workflow is a formal numerical run, even when selected
            # individually for development. Only quick smoke tests may proceed
            # with a version mismatch.
            check(False, f"version {name}", f"active={actual}, pinned={expected}",
                  warning=mode == "quick")

    try:
        import fdfi
        from fdfi.explainers import EOTExplainer, FlowExplainer, OTExplainer
        check(True, "fdfi import/version", fdfi.__version__)
        for cls, methods in ((EOTExplainer, ("__call__", "conf_int", "diagnose")),
                             (FlowExplainer, ("__call__", "explain_batches", "conf_int", "diagnose")),
                             (OTExplainer, ("__call__", "conf_int", "diagnose"))):
            missing = [name for name in methods if not callable(getattr(cls, name, None))]
            check(not missing, f"API {cls.__name__}", "OK" if not missing else f"missing {missing}")
    except Exception as exc:
        check(False, "fdfi import/API", repr(exc))

    common_files = {
        "CTG source notebook": CTG_NOTEBOOK,
        "sens50 source notebook": EOT_NOTEBOOK,
    }
    for label, path in common_files.items():
        check(path.is_file() and os.access(path, os.R_OK), label, relative(path))
    if "sens50" in selected:
        ok, detail = validate_sens50_inputs(
            CASE_STUDY_DATA_DIR / "sens50_processed_dataset.csv",
            CASE_STUDY_DATA_DIR / "feature_group.csv",
        )
        check(ok, "fixed sens50 inputs", detail)
    if "ctg" in selected:
        ok, detail = validate_ctg_input(ctg_path or DATA_DIR / "CTG.xls")
        check(ok, "CTG pinned input", detail)
    if "simulation" in selected:
        try:
            module = importlib.import_module("scripts.reproduce_simulation")
            sim_report = module.preflight(mode=mode)
            check(bool(sim_report.get("ready")), "manuscript simulation specification",
                  str(sim_report.get("detail", "no affirmative readiness evidence")))
        except Exception as exc:
            check(False, "manuscript simulation specification", repr(exc))

    parent = runs_dir if runs_dir.exists() else runs_dir.parent
    check(os.access(parent, os.W_OK), "run staging write permission", relative(runs_dir))
    for name in THREAD_ENV_VARS:
        check(os.environ.get(name) == "1", f"thread setting {name}",
              f"active={os.environ.get(name, 'UNSET')}, required=1",
              warning=mode == "quick" or workflow != "all")
    if emit and report.blockers:
        print("\nUnresolved blocking requirements:")
        for blocker in report.blockers:
            print(f"- {blocker}")
    return report


def _hardware_metadata() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.platform(), "python_executable": sys.executable,
        "python_version": platform.python_version(), "cpu_model": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(), "physical_cpu_count": None, "total_ram_bytes": None,
        "gpu_model": None, "cuda_version": None, "gpu_total_memory_bytes": None,
        "execution_device": "CPU", "torch_deterministic_algorithms": None,
    }
    try:
        import psutil
        info["physical_cpu_count"] = psutil.cpu_count(logical=False)
        info["total_ram_bytes"] = psutil.virtual_memory().total
    except ImportError:
        pass
    if sys.platform == "darwin":
        try:
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=True,
                                   capture_output=True, text=True).stdout.strip()
            if brand:
                info["cpu_model"] = brand
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        import torch
        info["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(gpu_model=props.name, gpu_total_memory_bytes=props.total_memory,
                        execution_device="CUDA")
    except ImportError:
        pass
    info["thread_environment"] = {name: os.environ.get(name, "UNSET") for name in
                                  ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
    return info


def _write_environment(config: RunConfig) -> Path:
    payload = _hardware_metadata()
    payload.update(run_id=config.run_id, mode=config.mode, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   git_commit=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain")))
    payload["dependencies"] = {name: _package_version(name) for name in
                               ("fdfi", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib",
                                "seaborn", "statsmodels", "openpyxl", "xlrd", "psutil", "tqdm", "pip",
                                "torch", "torchdiffeq")}
    try:
        import fdfi
        payload["dependencies"]["fdfi"] = fdfi.__version__
    except ImportError:
        pass
    path = config.metadata_dir / "environment.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def aggregate_status(results: Sequence[WorkflowResult]) -> Status:
    statuses = {result.status for result in results}
    if "FAILED" in statuses:
        return "FAILED"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    return "SUCCESS" if results and statuses == {"SUCCESS"} else "BLOCKED"


def _write_auxiliary_records(config: RunConfig, results: Sequence[WorkflowResult]) -> tuple[Path, Path]:
    runtime_path = config.metadata_dir / "runtime_memory.csv"
    with runtime_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["workflow", "status", "runtime_seconds", "stage_peak_cpu_rss_mb_sampled",
                  "cuda_peak_allocated_mb_stage"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for result in results:
            writer.writerow({"workflow": result.name, "status": result.status,
                             "runtime_seconds": f"{result.runtime_seconds:.6f}",
                             "stage_peak_cpu_rss_mb_sampled": result.process_peak_memory_mb if result.process_peak_memory_mb is not None else "UNAVAILABLE",
                             "cuda_peak_allocated_mb_stage": result.cuda_peak_allocated_mb if result.cuda_peak_allocated_mb is not None else "UNAVAILABLE"})
    key_path = config.tables_dir / "key_numerical_results.csv"
    with key_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["workflow", "metric", "value", "mode"]); writer.writeheader()
        for result in results:
            for metric, value in sorted(result.key_results.items()):
                writer.writerow({"workflow": result.name, "metric": metric, "value": value, "mode": config.mode})
    return runtime_path, key_path


def _artifact_entry(path: Path, source: str, config: RunConfig) -> dict[str, Any]:
    return {"path": relative(path), "run_relative_path": path.relative_to(config.output_root).as_posix(),
            "kind": path.suffix.lstrip(".") or "file", "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size, "source_workflow": source}


def _checksum_record(path: Path) -> dict[str, Any]:
    return ({"path": relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            if path.is_file() else {"path": relative(path), "sha256": None, "size_bytes": None})


def _run_provenance(workflow: WorkflowSelection, config: RunConfig) -> dict[str, Any]:
    selected = {item[0] for item in _selected(workflow)}
    data_paths = []
    if "sens50" in selected:
        data_paths += [CASE_STUDY_DATA_DIR / "sens50_processed_dataset.csv",
                       CASE_STUDY_DATA_DIR / "feature_group.csv"]
    if "ctg" in selected: data_paths += [DATA_DIR / "CTG.xls"]
    notebook_paths = [EOT_NOTEBOOK, CTG_NOTEBOOK]
    execution_paths = [
        REPLICATION_DIR / "reproduce.py", REPLICATION_DIR / "reproduce_quick.py",
        REPLICATION_DIR / "scripts/common.py", REPLICATION_DIR / "scripts/reproduce_sens50_eot.py",
        REPLICATION_DIR / "scripts/reproduce_ctg.py", REPLICATION_DIR / "scripts/reproduce_simulation.py",
        REPLICATION_DIR / "scripts/fetch_ctg.py", REPLICATION_DIR / "environment.yml",
        DATA_DIR / "checksums.sha256",
    ]
    settings = {}
    for path in sorted(config.metadata_dir.glob("*_settings.json")):
        settings[path.name] = json.loads(path.read_text(encoding="utf-8"))
    dirty_text = _git("status", "--porcelain")
    return {"input_data": [_checksum_record(path) for path in data_paths],
            "source_notebooks": [_checksum_record(path) for path in notebook_paths],
            "execution_files": [_checksum_record(path) for path in execution_paths],
            "workflow_scripts": [_checksum_record(path) for path in execution_paths if path.suffix == ".py"],
            "git": {"commit": _git("rev-parse", "HEAD"),
                    "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                    "dirty": bool(dirty_text), "dirty_files": dirty_text.splitlines() if dirty_text else []},
            "settings": settings}


def _write_manifest(config: RunConfig, results: Sequence[WorkflowResult], log_path: Path,
                    environment_path: Path, runtime_path: Path, key_path: Path,
                    workflow: WorkflowSelection, publication_attempted: bool,
                    publication_succeeded: bool, published_paths: Sequence[str]) -> Path:
    artifacts: list[dict[str, Any]] = []
    for result in results:
        for raw in result.generated_files:
            path = Path(raw)
            if not path.is_absolute():
                path = REPO_ROOT / path
            if path.is_file() and path.is_relative_to(config.output_root):
                artifacts.append(_artifact_entry(path, result.name, config))
    for path, source in ((key_path, "orchestrator"), (environment_path, "orchestrator"),
                         (runtime_path, "orchestrator"), (log_path, "orchestrator")):
        if path.is_file():
            artifacts.append(_artifact_entry(path, source, config))
    all_success = aggregate_status(results) == "SUCCESS"
    git_clean = not bool(_git("status", "--porcelain"))
    payload = {"run_id": config.run_id, "mode": config.mode, "workflow_selection": workflow,
               "formal_manuscript_run": bool(config.mode == "full" and workflow == "all" and
                                               all_success and publication_succeeded and git_clean),
               "formal_manuscript_run_reason": ("eligible" if config.mode == "full" and workflow == "all" and
                                                  all_success and publication_succeeded and git_clean else
                                                  "requires full/all success, successful publication, and clean Git worktree"),
               "publication_attempted": publication_attempted,
               "publication_succeeded": publication_succeeded,
               "published_stable_paths": list(published_paths),
               "status": aggregate_status(results), "workflows": [asdict(r) for r in results],
               "artifacts": artifacts, "provenance": _run_provenance(workflow, config)}
    path = config.metadata_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _publish_formal(config: RunConfig, figures_dir: Path = FIGURES_DIR,
                    tables_dir: Path = TABLES_DIR, results_dir: Path = RESULTS_DIR,
                    state_path: Path | None = None) -> list[str]:
    """Transactionally publish a complete full run without touching unrelated files."""
    prepared: list[tuple[Path, Path, Path | None]] = []
    state_path = state_path or REPLICATION_DIR / "metadata/formal_publication.json"
    previous_paths: list[str] = []
    if state_path.is_file():
        previous_paths = json.loads(state_path.read_text(encoding="utf-8")).get("managed_paths", [])
    for source_dir, destination_dir in ((config.figures_dir, figures_dir),
                                        (config.tables_dir, tables_dir),
                                        (config.results_dir, results_dir)):
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.iterdir():
            if source.is_file():
                target = destination_dir / source.name
                temporary = destination_dir / f".{source.name}.{config.run_id}.part"
                try:
                    shutil.copy2(source, temporary)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    for prior, _, _ in prepared:
                        prior.unlink(missing_ok=True)
                    raise
                backup = destination_dir / f".{source.name}.{config.run_id}.backup" if target.exists() else None
                prepared.append((temporary, target, backup))
    new_targets = {target.resolve() for _, target, _ in prepared}
    managed_roots = (figures_dir.resolve(), tables_dir.resolve(), results_dir.resolve())
    stale: list[tuple[Path, Path]] = []
    for raw in previous_paths:
        target = REPO_ROOT / raw
        resolved = target.resolve()
        if resolved not in new_targets and any(resolved.is_relative_to(root) for root in managed_roots) and target.is_file():
            stale.append((target, target.with_name(f".{target.name}.{config.run_id}.stale-backup")))
    committed: list[tuple[Path, Path | None]] = []
    try:
        for target, backup in stale:
            os.replace(target, backup)
        for temporary, target, backup in prepared:
            if backup is not None:
                os.replace(target, backup)
            try:
                os.replace(temporary, target)
            except Exception:
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                raise
            committed.append((target, backup))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_tmp = state_path.with_name(f".{state_path.name}.{config.run_id}.part")
        state_tmp.write_text(json.dumps({"run_id": config.run_id,
                                         "managed_paths": [relative(target) for target, _ in committed]}, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(state_tmp, state_path)
    except Exception:
        for target, backup in reversed(committed):
            target.unlink(missing_ok=True)
            if backup is not None and backup.exists():
                os.replace(backup, target)
        for temporary, _, backup in prepared:
            temporary.unlink(missing_ok=True)
        for target, backup in stale:
            if backup.exists(): os.replace(backup, target)
        raise
    for _, backup in committed:
        if backup is not None:
            backup.unlink(missing_ok=True)
    for _, backup in stale:
        backup.unlink(missing_ok=True)
    return [relative(target) for target, _ in committed]


def _invoke_preflight(function: Callable[..., PreflightReport], mode: Mode,
                      workflow: WorkflowSelection, emit: bool,
                      runs_dir: Path) -> PreflightReport:
    parameters = inspect.signature(function).parameters.values()
    accepts_kwargs = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters)
    names = {item.name for item in parameters}
    options: dict[str, Any] = {}
    if accepts_kwargs or "emit" in names:
        options["emit"] = emit
    if accepts_kwargs or "runs_dir" in names:
        options["runs_dir"] = runs_dir
    return function(mode, workflow, **options)


def run_replication(mode: Mode, check_only: bool = False, workflow: WorkflowSelection = "all",
                    *, workflow_defs: Sequence[tuple[str, str, str]] | None = None,
                    preflight_fn: Callable[..., PreflightReport] = preflight_check,
                    publish_fn: Callable[[RunConfig], list[str]] = _publish_formal,
                    runs_dir: Path = RUNS_DIR) -> int:
    if check_only:
        return 0 if _invoke_preflight(preflight_fn, mode, workflow, True, runs_dir).ready else 1
    # Every full run is strict: selected workflows are useful for incremental
    # validation, but their numerical results still require the notebook-aligned
    # environment and validated inputs.
    report = _invoke_preflight(preflight_fn, mode, workflow, False, runs_dir)
    if mode == "full" and not report.ready:
        _invoke_preflight(preflight_fn, mode, workflow, True, runs_dir)
        print("Strict full preflight failed; no workflows started and no outputs modified.")
        return 2

    config = RunConfig.create(mode, runs_dir=runs_dir)
    config.ensure_directories()
    log_path = config.logs_dir / f"{config.run_id}.log"
    results: list[WorkflowResult] = []
    definitions = tuple(workflow_defs) if workflow_defs is not None else _selected(workflow)
    with log_path.open("w", encoding="utf-8") as log_handle:
        tee = _Tee(sys.stdout, log_handle)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"FDFI JSS replication\nRun ID: {config.run_id}\nMode: {mode}; workflow: {workflow}")
            print(f"Started: {datetime.now(timezone.utc).isoformat()}")
            print(f"Configuration: staging={relative(config.output_root)} seed={config.seed}")
            _invoke_preflight(preflight_fn, mode, workflow, True, runs_dir)
            set_random_seeds(config.seed)
            environment_path = _write_environment(config)
            prior_required_stop = False
            for _, label, module_name in definitions:
                if prior_required_stop:
                    results.append(WorkflowResult(label, "SKIPPED", error="prior required stage did not complete"))
                    continue
                print(f"\n--- {label} ---")
                started = datetime.now(timezone.utc).isoformat(); before = time.perf_counter()
                rss_monitor = PeakRSSMonitor(); rss_monitor.start()
                cuda_peak_allocated_mb(reset=True)
                try:
                    module = importlib.import_module(module_name)
                    result = module.run(mode=mode, config=config)
                    if not isinstance(result, WorkflowResult):
                        raise TypeError(f"{module_name}.run did not return WorkflowResult")
                except WorkflowBlocked as exc:
                    result = WorkflowResult(label, "BLOCKED", started_at=started,
                                            ended_at=datetime.now(timezone.utc).isoformat(),
                                            runtime_seconds=time.perf_counter() - before,
                                            error=str(exc))
                    print(f"BLOCKED: {exc}\n{traceback.format_exc()}")
                except Exception as exc:
                    result = WorkflowResult(label, "FAILED", started_at=started,
                                            ended_at=datetime.now(timezone.utc).isoformat(),
                                            runtime_seconds=time.perf_counter() - before,
                                            error=repr(exc))
                    print(f"FAILED: {exc}\n{traceback.format_exc()}")
                result.process_peak_memory_mb = rss_monitor.stop_mb()
                result.cuda_peak_allocated_mb = cuda_peak_allocated_mb()
                results.append(result)
                print(f"Stage status: {result.status}; runtime={result.runtime_seconds:.3f}s; seeds={result.seeds}; warnings={result.warnings}")
                if result.status != "SUCCESS":
                    prior_required_stop = True

            status = aggregate_status(results)
            published: list[str] = []
            publication_attempted = False
            publication_succeeded = False
            # These files must exist in staging before formal publication.
            runtime_path, key_path = _write_auxiliary_records(config, results)
            if status == "SUCCESS" and mode == "full" and workflow == "all":
                publication_attempted = True
                try:
                    published = publish_fn(config)
                    publication_succeeded = True
                except Exception as exc:
                    results.append(WorkflowResult("formal artifact publication", "FAILED", error=repr(exc)))
                    status = "FAILED"
                    print(f"FAILED during formal publication: {exc}\n{traceback.format_exc()}")
            print("\nFinal status summary:")
            for result in results:
                print(f"- {result.name}: {result.status}")
            print(f"Overall: {status}\nEnded: {datetime.now(timezone.utc).isoformat()}")
            if status == "SUCCESS" and mode == "full" and workflow == "all":
                print(f"Published formal artifacts: {published}")
            elif mode == "full":
                print("Formal outputs were not published.")
            log_handle.flush()
            manifest_path = config.metadata_dir / "run_manifest.json"
            print(f"Run manifest: {relative(manifest_path)}")
            log_handle.flush()
            _write_manifest(config, results, log_path, environment_path, runtime_path, key_path,
                            workflow, publication_attempted, publication_succeeded, published)
    return 0 if aggregate_status(results) == "SUCCESS" else 2
