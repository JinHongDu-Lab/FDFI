"""Show editorial points 3--6 using small synthetic data and local source.

Run: python replication/case_study/check_review_points.py
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import fdfi
from fdfi.explainers import Crossfitting, EOTExplainer, OTExplainer


def prediction(X):
    return 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] ** 2


def make_explainer(name, background):
    options = dict(nsamples=10, random_state=0, compute_diagnostics=False)
    if name == "ot":
        return OTExplainer(prediction, background, **options)
    if name == "eot":
        return EOTExplainer(prediction, background, epsilon=0.1, n_eta=4, **options)
    if name == "crossfitting":
        return Crossfitting(prediction, background, cv=3, **options)

    import torch
    from fdfi.explainers import FlowExplainer

    torch.manual_seed(0)
    torch.set_num_threads(1)
    return FlowExplainer(
        prediction, background, num_steps=20, hidden_dim=16,
        time_embed_dim=8, device=torch.device("cpu"), verbose=False, **options
    )


def show(code, value):
    print(f">>> {code}")
    print(value)


def demonstrate(name, background, X_test, y_test):
    explainer = make_explainer(name, background)
    before = str(explainer)
    print(f"\n{'=' * 72}\n{type(explainer).__name__}\n{'=' * 72}")

    print("\n# Point 3: Explanation results")
    print(">>> results = explainer(X_test, y=y_test)")
    results = explainer(X_test, y=y_test)
    show("type(results)", type(results))
    for field in ("phi_X", "phi_Z", "se_X", "se_Z", "std_X", "std_Z"):
        show(f"results.get_{field}()", getattr(results, f"get_{field}")())
    show("results.top_features(k=3)", results.top_features(k=3))
    show("results.to_frame()", results.to_frame().to_string())

    print("\n# Point 4: Inference results")
    print('>>> ci = explainer.conf_int(multitest_method="fdr_bh")')
    ci = explainer.conf_int(multitest_method="fdr_bh")
    show("type(ci)", type(ci))
    for field in ("score", "se", "ci", "pvalue", "pvalue_adj", "ranking"):
        show(f"ci.get_{field}()", getattr(ci, f"get_{field}")())
    show("ci.significant()", ci.significant())
    show("ci.to_frame()", ci.to_frame().to_string())

    print("\n# Point 5: Object printing")
    show("print(explainer)  # Before scoring (saved output)", before)
    show("print(explainer)  # After scoring", explainer)
    show("print(results)", results)
    show("print(ci)", ci)
    if name == "flow":
        show("print(explainer.flow_model)", explainer.flow_model)

    print("\n# Point 6: Summary output")
    print(">>> returned = explainer.summary()")
    returned = explainer.summary()
    show("returned", returned)
    print(">>> text = explainer.summary(print_output=False)")
    text = explainer.summary(print_output=False)
    show("type(text)", type(text))
    show("print(text)", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--explainers", nargs="+", choices=("ot", "eot", "flow", "crossfitting"),
        default=["ot", "eot", "flow", "crossfitting"],
    )
    parser.add_argument(
        "--log", type=Path,
        default=Path(__file__).resolve().parent / "logs" / "review_points_3_4_5_6.log",
    )
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4))
    X[:, 1] = 0.5 * X[:, 0] + np.sqrt(0.75) * X[:, 1]
    y = prediction(X) + rng.normal(0, 0.2, len(X))
    np.set_printoptions(precision=6, suppress=True)

    with args.log.open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            print(f"fdfi {fdfi.__version__}: {fdfi.__file__}")
            print("Synthetic data: 80 background rows, 40 test rows, 4 features.")
            print("Flow: 20 training steps for the API demo.")
            for name in args.explainers:
                demonstrate(name, X[:80], X[80:], y[80:])
    print(args.log.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
