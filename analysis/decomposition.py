from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from analysis.statistics import add_effect_fdr, cluster_bootstrap, cluster_ids


CONDITIONS = ("O", "L", "P", "F")
PRIMARY_EFFECTS = ("L_minus_P", "O_minus_L")
AUXILIARY_EFFECTS = ("L_minus_F", "P_minus_F", "interaction")
ALL_EFFECTS = (*PRIMARY_EFFECTS, *AUXILIARY_EFFECTS)


def safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=int)
    if np.unique(labels_array).size != 2:
        return float("nan")
    return float(roc_auc_score(labels_array, np.asarray(scores, dtype=float)))


def pivot_scores(frame: pd.DataFrame) -> pd.DataFrame:
    index = ["dataset", "id", "label", "speaker"]
    wide = frame.pivot(index=index, columns="condition", values="score").reset_index()
    missing = set(CONDITIONS) - set(wide.columns)
    if missing:
        raise ValueError(f"score table is missing conditions {sorted(missing)}")
    if wide[list(CONDITIONS)].isna().any().any():
        raise ValueError("condition scores are incomplete after pivot")
    return wide


def auc_decomposition(wide: pd.DataFrame) -> dict[str, float]:
    aucs = {
        f"auc_{condition}": safe_auc(wide["label"], wide[condition])
        for condition in CONDITIONS
    }
    return {
        **aucs,
        "L_minus_P": aucs["auc_L"] - aucs["auc_P"],
        "O_minus_L": aucs["auc_O"] - aucs["auc_L"],
        "L_minus_F": aucs["auc_L"] - aucs["auc_F"],
        "P_minus_F": aucs["auc_P"] - aucs["auc_F"],
        "interaction": aucs["auc_O"] - aucs["auc_L"] - aucs["auc_P"] + aucs["auc_F"],
    }


def build_effect_table(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(list(group_columns), sort=True, dropna=False)
    for index, (group_values, group) in enumerate(grouped):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        wide = pivot_scores(group)
        row: dict[str, object] = dict(zip(group_columns, group_values))
        row.update(
            {
                "n_utterances": len(wide),
                "n_speakers": wide["speaker"].nunique(),
                "n_clusters": cluster_ids(wide).nunique(),
            }
        )
        row.update(auc_decomposition(wide))
        intervals, p_values = cluster_bootstrap(
            wide,
            statistic=auc_decomposition,
            replicates=bootstrap_replicates,
            seed=seed + index,
        )
        for name, (lower, upper) in intervals.items():
            row[f"{name}_ci_lower"] = lower
            row[f"{name}_ci_upper"] = upper
        for name in ALL_EFFECTS:
            row[f"{name}_p_value"] = p_values[name]
        rows.append(row)
    return add_effect_fdr(pd.DataFrame(rows), ALL_EFFECTS)


def compare_score_methods(
    single_token: pd.DataFrame, token_set: pd.DataFrame
) -> pd.DataFrame:
    keys = ["setting", "dataset", "id", "condition"]
    single = single_token.loc[single_token["condition"].isin(CONDITIONS)]
    token = token_set.loc[token_set["condition"].isin(CONDITIONS)]
    common = single.merge(
        token,
        on=keys,
        how="inner",
        suffixes=("_single_token", "_token_set"),
        validate="one_to_one",
    )
    if common["label_single_token"].ne(common["label_token_set"]).any():
        raise ValueError("score-method inputs disagree on labels")
    if len(common) != len(single):
        raise ValueError(f"score-method join matched {len(common)} rows, expected {len(single)}")
    rows: list[dict[str, object]] = []
    for (setting, dataset, condition), group in common.groupby(
        ["setting", "dataset", "condition"], sort=True
    ):
        labels = group["label_single_token"]
        single_auc = safe_auc(labels, group["score_single_token"])
        token_auc = safe_auc(labels, group["score_token_set"])
        rows.append(
            {
                "setting": setting,
                "dataset": dataset,
                "condition": condition,
                "n": len(group),
                "spearman": float(
                    spearmanr(
                        group["score_single_token"], group["score_token_set"]
                    ).statistic
                ),
                "auc_single_token": single_auc,
                "auc_token_set": token_auc,
                "auc_difference_single_minus_set": single_auc - token_auc,
                "sd_single_token": float(group["score_single_token"].std()),
                "sd_token_set": float(group["score_token_set"].std()),
            }
        )
    return pd.DataFrame(rows)


def assess_h4(full_effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "setting",
        "dataset",
        "n_utterances",
        "auc_F",
        "auc_F_ci_lower",
        "auc_F_ci_upper",
        "P_minus_F",
        "L_minus_P",
    ]
    missing = set(columns) - set(full_effects.columns)
    if missing:
        raise ValueError(f"H4 assessment missing columns {sorted(missing)}")
    cells = full_effects[columns].copy()
    cells["f_floor_compatible"] = (
        cells["auc_F_ci_lower"].le(0.5) & cells["auc_F_ci_upper"].ge(0.5)
    )
    cells["interpretation"] = np.where(
        cells["f_floor_compatible"],
        "F interval includes the random floor",
        "F interval excludes the random floor; P-F is not comparable across settings",
    )
    decisions: list[dict[str, object]] = []
    for dataset, group in cells.groupby("dataset", sort=True):
        effects = group.set_index("setting")["P_minus_F"].to_dict()
        valid = set(group["setting"]) == {"target_only", "audio_plus_context"}
        valid = valid and bool(group["f_floor_compatible"].all())
        decisions.append(
            {
                "dataset": dataset,
                "target_P_minus_F": effects.get("target_only", np.nan),
                "context_P_minus_F": effects.get("audio_plus_context", np.nan),
                "comparison_valid": valid,
                "decision": (
                    "H4 testable under the F-baseline diagnostic"
                    if valid
                    else "H4 not identifiable under the current F-baseline design"
                ),
            }
        )
    return cells, pd.DataFrame(decisions)

