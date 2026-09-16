from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.decomposition import (
    AUXILIARY_EFFECTS,
    PRIMARY_EFFECTS,
    assess_h4,
    build_effect_table,
    compare_score_methods,
)
from analysis.statistics import fit_mixed_effects
from analysis.visualization import generate_figures
from preprocessing.validation import (
    load_full_single_token_predictions,
    load_full_token_set_predictions,
    load_seven_model_predictions,
)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def choose_path(cli_value: Path | None, config_value: object, name: str) -> Path:
    raw = cli_value if cli_value is not None else config_value
    if raw is None or not str(raw).strip():
        raise ValueError(f"{name} must be set in the config or on the command line")
    return Path(str(raw)).expanduser().resolve()


def prepare_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run_analysis(
    results_root: Path,
    token_set_root: Path,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
    render_figures: bool = True,
) -> dict[str, object]:
    prepare_output_directory(output_dir)
    single_token, single_sources = load_full_single_token_predictions(results_root)
    token_set, token_set_sources = load_full_token_set_predictions(token_set_root)
    seven_model, seven_model_sources = load_seven_model_predictions(results_root)

    comparison = compare_score_methods(single_token, token_set)
    single_effects = build_effect_table(
        single_token,
        ("model", "setting", "dataset"),
        bootstrap_replicates,
        seed,
    )
    token_primary = token_set.loc[token_set["condition"].isin(("O", "L", "P", "F"))]
    token_effects = build_effect_table(
        token_primary,
        ("model", "setting", "dataset"),
        bootstrap_replicates,
        seed + 10_000,
    )
    seven_effects = build_effect_table(
        seven_model,
        ("model", "setting", "dataset"),
        bootstrap_replicates,
        seed + 20_000,
    )
    h4_cells, h4_decisions = assess_h4(token_effects)
    mixed_omnibus, mixed_by_dataset = fit_mixed_effects(seven_model)

    output_files = {
        "score_method_comparison": "score_method_comparison.csv",
        "full_single_token_auc_effects": "full_single_token_auc_effects.csv",
        "full_token_set_auc_effects": "full_token_set_auc_effects.csv",
        "seven_model_auc_effects": "seven_model_auc_effects.csv",
        "h4_f_baseline_assessment": "h4_f_baseline_assessment.csv",
        "h4_decision": "h4_decision.csv",
        "mixed_effects_omnibus": "mixed_effects_omnibus.csv",
        "mixed_effects_by_dataset": "mixed_effects_by_dataset.csv",
        "audit": "analysis_audit.json",
    }
    tables = {
        "score_method_comparison": comparison,
        "full_single_token_auc_effects": single_effects,
        "full_token_set_auc_effects": token_effects,
        "seven_model_auc_effects": seven_effects,
        "h4_f_baseline_assessment": h4_cells,
        "h4_decision": h4_decisions,
        "mixed_effects_omnibus": mixed_omnibus,
        "mixed_effects_by_dataset": mixed_by_dataset,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / output_files[name], index=False)

    figure_paths = generate_figures(output_dir) if render_figures else []
    mixed_error = bool(
        mixed_omnibus["term"].eq("**MODEL_ERROR**").any()
        or mixed_by_dataset["term"].eq("**MODEL_ERROR**").any()
    )
    audit: dict[str, object] = {
        "schema_version": "1.1",
        "status": "MODEL_ERROR" if mixed_error else "OK",
        "primary_metric": "AUROC",
        "primary_score_method": "token_set_logsumexp",
        "sensitivity_score_method": "single_token",
        "primary_effects": list(PRIMARY_EFFECTS),
        "auxiliary_f_baseline_effects": list(AUXILIARY_EFFECTS),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "cmma_unit": "conversation parsed from utterance ID; speaker fallback",
            "mustard_unit": "speaker",
            "multiple_testing": "Benjamini-Hochberg FDR across reported effects",
        },
        "counts": {
            "full_single_token_rows": len(single_token),
            "full_token_set_rows": len(token_set),
            "seven_model_rows": len(seven_model),
            "seven_models": seven_model["model"].nunique(),
            "score_comparison_cells": len(comparison),
            "mixed_effects_omnibus_terms": len(mixed_omnibus),
            "mixed_effects_by_dataset_terms": len(mixed_by_dataset),
        },
        "mixed_effects": {
            "outcome": "within-model standardized label-adjusted margin",
            "reference_condition": "O",
            "random_intercept": "dataset::speaker",
            "multiple_testing": "Benjamini-Hochberg FDR within each model family",
            "model_error": mixed_error,
        },
        "h4_decisions": h4_decisions.to_dict(orient="records"),
        "inputs": {
            "results_root": str(results_root),
            "token_set_root": str(token_set_root),
            "single_token": single_sources,
            "token_set": token_set_sources,
            "seven_model": seven_model_sources,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "outputs": {
            **output_files,
            "figures": [path.name for path in figure_paths],
        },
    }
    (output_dir / output_files["audit"]).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-sample context probes and seven-model analyses."
    )
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml"
    )
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--token-set-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    analysis = config["analysis"]
    paths = config["paths"]
    replicates = (
        args.bootstrap_replicates
        if args.bootstrap_replicates is not None
        else int(analysis["bootstrap_replicates"])
    )
    audit = run_analysis(
        results_root=choose_path(args.results_root, paths.get("results_root"), "results_root"),
        token_set_root=choose_path(
            args.token_set_root, paths.get("token_set_root"), "token_set_root"
        ),
        output_dir=choose_path(args.output_dir, paths.get("output_dir"), "output_dir"),
        bootstrap_replicates=replicates,
        seed=args.seed if args.seed is not None else int(analysis["seed"]),
        render_figures=not args.no_figures,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
