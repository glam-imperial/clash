from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


CONDITIONS = ("O", "L", "P", "F")


def adjust_bh(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    ordered = numeric.loc[numeric.notna()].sort_values()
    if ordered.empty:
        return result
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    adjusted = ordered.to_numpy(dtype=float) * len(ordered) / ranks
    adjusted = np.minimum(1.0, np.minimum.accumulate(adjusted[::-1])[::-1])
    result.loc[ordered.index] = adjusted
    return result


def cluster_ids(wide: pd.DataFrame) -> pd.Series:
    cluster = wide["speaker"].astype(str).copy()
    cmma = wide["dataset"].eq("CMMA")
    conversation = wide.loc[cmma, "id"].astype(str).str.extract(
        r"^(C_[0-9]+)_U_", expand=False
    )
    cluster.loc[cmma] = conversation.fillna(cluster.loc[cmma])
    return wide["dataset"].astype(str) + "::" + cluster


def cluster_bootstrap(
    wide: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], dict[str, float]],
    replicates: int,
    seed: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    if replicates < 100:
        raise ValueError("cluster bootstrap requires at least 100 replicates")
    rng = np.random.default_rng(seed)
    cluster = cluster_ids(wide).to_numpy()
    groups = [np.flatnonzero(cluster == value) for value in pd.unique(cluster)]
    if len(groups) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    draws: dict[str, list[float]] = {}
    for _ in range(replicates):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        for name, value in statistic(wide.iloc[indices]).items():
            draws.setdefault(name, []).append(value)

    intervals: dict[str, tuple[float, float]] = {}
    p_values: dict[str, float] = {}
    for name, values in draws.items():
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if len(array) < max(20, replicates // 2):
            intervals[name] = (float("nan"), float("nan"))
            p_values[name] = float("nan")
            continue
        intervals[name] = (
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        )
        lower_tail = (np.count_nonzero(array <= 0) + 1) / (len(array) + 1)
        upper_tail = (np.count_nonzero(array >= 0) + 1) / (len(array) + 1)
        p_values[name] = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return intervals, p_values


def add_effect_fdr(
    table: pd.DataFrame,
    effects: Sequence[str],
) -> pd.DataFrame:
    result = table.copy()
    p_columns = [f"{effect}_p_value" for effect in effects]
    stacked = result[p_columns].stack().dropna()
    adjusted = adjust_bh(stacked)
    for effect in effects:
        result[f"{effect}_p_value_bh_fdr"] = np.nan
    for (row_index, column), value in adjusted.items():
        effect = column.removesuffix("_p_value")
        result.loc[row_index, f"{effect}_p_value_bh_fdr"] = value
    return result


def prepare_mixed_effects_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.loc[frame["condition"].isin(CONDITIONS)].copy()
    data["adjusted_margin"] = pd.to_numeric(data["margin"], errors="raise") * np.where(
        data["label"].eq(1), 1.0, -1.0
    )
    means = data.groupby("model")["adjusted_margin"].transform("mean")
    scales = data.groupby("model")["adjusted_margin"].transform(
        lambda values: values.std(ddof=0)
    )
    if scales.isna().any() or scales.eq(0).any():
        raise ValueError("cannot standardize a model with constant or missing margins")
    data["standardized_adjusted_margin"] = (data["adjusted_margin"] - means) / scales
    data["condition"] = pd.Categorical(data["condition"], categories=CONDITIONS, ordered=True)
    data["cluster"] = data["dataset"].astype(str) + "::" + data["speaker"].astype(str)
    return data


def _fit_mixed_model(
    data: pd.DataFrame, formula: str, analysis: str, dataset: str
) -> pd.DataFrame:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(formula, data=data, groups=data["cluster"], re_formula="1")
            fitted = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        warning_text = " | ".join(dict.fromkeys(str(item.message) for item in caught))
        return pd.DataFrame(
            {
                "analysis": analysis,
                "dataset": dataset,
                "term": fitted.params.index,
                "estimate": fitted.params.to_numpy(dtype=float),
                "std_error": fitted.bse.to_numpy(dtype=float),
                "z_value": fitted.tvalues.to_numpy(dtype=float),
                "p_value": fitted.pvalues.to_numpy(dtype=float),
                "n_rows": len(data),
                "n_clusters": data["cluster"].nunique(),
                "converged": bool(fitted.converged),
                "formula": formula,
                "outcome": "within-model standardized label-adjusted margin",
                "fit_warnings": warning_text,
            }
        )
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "analysis": analysis,
                    "dataset": dataset,
                    "term": "**MODEL_ERROR**",
                    "n_rows": len(data),
                    "n_clusters": data["cluster"].nunique(),
                    "converged": False,
                    "formula": formula,
                    "outcome": "within-model standardized label-adjusted margin",
                    "error": repr(exc),
                }
            ]
        )


def _add_model_fdr(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result["p_value_bh_fdr"] = np.nan
    if "p_value" not in result.columns:
        return result
    fixed = ~result["term"].isin(("Intercept", "Group Var"))
    fixed &= ~result["term"].astype(str).str.startswith("**")
    if group_columns:
        groups = result.loc[fixed].groupby(group_columns, dropna=False).groups.values()
        for indices in groups:
            result.loc[indices, "p_value_bh_fdr"] = adjust_bh(result.loc[indices, "p_value"])
    else:
        result.loc[fixed, "p_value_bh_fdr"] = adjust_bh(result.loc[fixed, "p_value"])
    return result


def fit_mixed_effects(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = prepare_mixed_effects_data(frame)
    omnibus_formula = (
        "standardized_adjusted_margin ~ C(model) * "
        "C(condition, Treatment(reference='O')) + C(dataset)"
    )
    omnibus = _add_model_fdr(
        _fit_mixed_model(data, omnibus_formula, "omnibus", "all"), []
    )
    detail_formula = (
        "standardized_adjusted_margin ~ C(model) * "
        "C(condition, Treatment(reference='O'))"
    )
    detail = pd.concat(
        [
            _fit_mixed_model(subset, detail_formula, "by_dataset", str(dataset))
            for dataset, subset in data.groupby("dataset", sort=True, observed=True)
        ],
        ignore_index=True,
    )
    detail = _add_model_fdr(detail, ["dataset"])
    detail["inference_note"] = np.where(
        detail["n_clusters"].lt(5),
        "exploratory: fewer than five speaker clusters",
        "",
    )
    return omnibus, detail
