from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from preprocessing.world_processing import analyse_world, load_audio


CONDITIONS = ("O", "L", "P", "F")
COMMON_KEYS = ("dataset", "id", "condition")
EXPECTED_FULL_UTTERANCES = 4625
EXPECTED_HOLDOUT_UTTERANCES = 944
EXPECTED_SEVEN_MODELS = 7


def require_columns(
    frame: pd.DataFrame, columns: Iterable[str], source: Path | str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}")


def first_numeric(
    frame: pd.DataFrame, candidates: Iterable[str]
) -> tuple[pd.Series, str]:
    for column in candidates:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values, column
    raise ValueError(f"no usable numeric score among {list(candidates)}")


def coalesce_numeric(
    frame: pd.DataFrame, candidates: Iterable[str]
) -> tuple[pd.Series, str]:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    used: list[str] = []
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        fill = result.isna() & values.notna()
        result.loc[fill] = values.loc[fill]
        if fill.any():
            used.append(column)
    if result.isna().any():
        raise ValueError(f"missing scores after coalescing {list(candidates)}")
    return result, "+".join(used)


def load_metadata(root: Path, prefer_full: bool) -> tuple[pd.DataFrame, Path]:
    names = (
        ("all.csv", "clash_metadata_final.csv", "test_all.csv")
        if prefer_full
        else ("test_all.csv", "clash_metadata_final.csv", "all.csv")
    )
    candidates = [root / "metadata" / name for name in names]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"metadata not found; expected one of {[str(item) for item in candidates]}"
        )
    frame = pd.read_csv(path, dtype={"id": str})
    require_columns(frame, ("dataset", "id", "label", "speaker"), path)
    if "condition" in frame.columns:
        frame = frame.loc[frame["condition"].eq("O")].copy()
    frame = frame[["dataset", "id", "label", "speaker"]].drop_duplicates()
    if frame[["dataset", "id"]].duplicated().any():
        raise ValueError(f"{path}: duplicate metadata keys")
    frame["id"] = frame["id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not frame["label"].isin((0, 1)).all():
        raise ValueError(f"{path}: labels must be binary 0/1")
    frame["speaker"] = frame["speaker"].astype("string")
    if frame.isna().any().any():
        raise ValueError(f"{path}: metadata contains missing values")
    return frame, path


def attach_metadata(
    predictions: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    require_columns(predictions, COMMON_KEYS, "prediction table")
    frame = predictions.copy()
    frame["id"] = frame["id"].astype(str)
    attached = frame.merge(
        metadata.rename(
            columns={"label": "metadata_label", "speaker": "metadata_speaker"}
        ),
        on=["dataset", "id"],
        how="left",
        validate="many_to_one",
    )
    if "label" in attached.columns:
        attached["label"] = pd.to_numeric(attached["label"], errors="raise").astype(int)
        mismatch = attached["label"].ne(attached["metadata_label"])
        if mismatch.any():
            raise ValueError(
                f"prediction labels disagree with metadata in {int(mismatch.sum())} rows"
            )
    else:
        attached["label"] = attached["metadata_label"]
    if "speaker" in attached.columns:
        attached["speaker"] = attached["speaker"].fillna(attached["metadata_speaker"])
    else:
        attached["speaker"] = attached["metadata_speaker"]
    attached = attached.drop(columns=["metadata_label", "metadata_speaker"])
    if attached[["label", "speaker"]].isna().any().any():
        raise ValueError("metadata attachment left missing labels or speakers")
    attached["label"] = attached["label"].astype(int)
    return attached


def validate_prediction_keys(
    frame: pd.DataFrame,
    name: str,
    key_prefix: Iterable[str],
    expected_conditions: Iterable[str],
) -> None:
    keys = [*key_prefix, *COMMON_KEYS]
    if frame[keys].duplicated().any():
        raise ValueError(f"{name}: duplicate keys {keys}")
    missing = set(expected_conditions) - set(frame["condition"].astype(str))
    if missing:
        raise ValueError(f"{name}: missing conditions {sorted(missing)}")


def load_full_single_token_predictions(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    metadata, metadata_path = load_metadata(root, prefer_full=True)
    sources = (
        (root / "qwen3" / "qwen3_token_logits_full.csv", "target_only"),
        (
            root / "qwen3" / "qwen3_audio_plus_context_token_logits_full.csv",
            "audio_plus_context",
        ),
    )
    parts: list[pd.DataFrame] = []
    provenance = {"metadata": str(metadata_path)}
    for path, setting in sources:
        frame = pd.read_csv(path, dtype={"id": str})
        require_columns(frame, COMMON_KEYS, path)
        if "error" in frame.columns:
            frame = frame.loc[frame["error"].isna()].copy()
        score, score_column = first_numeric(frame, ("score", "token_margin", "margin"))
        margin, margin_column = first_numeric(
            frame, ("token_margin", "margin", "logit_score", "score")
        )
        part = frame[["dataset", "id", "condition"]].copy()
        part["score"] = score.to_numpy()
        part["margin"] = margin.to_numpy()
        part["setting"] = setting
        part["model"] = "qwen3_omni_30b"
        part["score_method"] = "single_token"
        parts.append(attach_metadata(part, metadata))
        provenance[setting] = f"{path};score={score_column};margin={margin_column}"
    result = pd.concat(parts, ignore_index=True)
    validate_prediction_keys(result, "full single-token predictions", ("setting",), CONDITIONS)
    expected = 2 * EXPECTED_FULL_UTTERANCES * len(CONDITIONS)
    if len(result) != expected:
        raise ValueError(f"single-token rows={len(result)}, expected={expected}")
    return result, provenance


def load_full_token_set_predictions(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    metadata, metadata_path = load_metadata(root, prefer_full=True)
    sources = (
        (
            root / "predictions" / "prediction_qwen3omni_target_logprob.csv",
            "target_only",
        ),
        (
            root / "predictions" / "prediction_qwen3omni_context_logprob.csv",
            "audio_plus_context",
        ),
    )
    parts: list[pd.DataFrame] = []
    provenance = {"metadata": str(metadata_path)}
    for path, setting in sources:
        frame = pd.read_csv(path, dtype={"sample_id": str})
        require_columns(frame, ("sample_id", "dataset", "condition", "logit"), path)
        part = frame.rename(columns={"sample_id": "id"})[
            ["dataset", "id", "condition", "logit"]
        ].copy()
        part["score"] = pd.to_numeric(part.pop("logit"), errors="raise")
        part["margin"] = part["score"]
        part["setting"] = setting
        part["model"] = "qwen3_omni_30b"
        part["score_method"] = "token_set_logsumexp"
        parts.append(attach_metadata(part, metadata))
        provenance[setting] = str(path)
    result = pd.concat(parts, ignore_index=True)
    expected_conditions = (*CONDITIONS, "L_TTS")
    validate_prediction_keys(result, "full token-set predictions", ("setting",), expected_conditions)
    expected = 2 * EXPECTED_FULL_UTTERANCES * len(expected_conditions)
    if len(result) != expected:
        raise ValueError(f"token-set rows={len(result)}, expected={expected}")
    return result, provenance


def load_seven_model_predictions(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    path = root / "analysis" / "audio_7models" / "predictions_7models.csv"
    frame = pd.read_csv(path, dtype={"id": str})
    require_columns(frame, ("model", *COMMON_KEYS), path)
    score, score_columns = coalesce_numeric(
        frame, ("score", "margin", "logit_score", "token_margin")
    )
    margin, margin_columns = coalesce_numeric(
        frame, ("margin", "logit_score", "token_margin", "score")
    )
    frame["score"] = score
    frame["margin"] = margin
    frame["setting"] = "target_only"
    metadata, metadata_path = load_metadata(root, prefer_full=False)
    frame = attach_metadata(frame, metadata)
    validate_prediction_keys(frame, "seven-model predictions", ("model",), CONDITIONS)
    expected = EXPECTED_SEVEN_MODELS * EXPECTED_HOLDOUT_UTTERANCES * len(CONDITIONS)
    if frame["model"].nunique() != EXPECTED_SEVEN_MODELS or len(frame) != expected:
        raise ValueError(
            f"seven-model rows={len(frame)}, models={frame['model'].nunique()}; "
            f"expected rows={expected}, models={EXPECTED_SEVEN_MODELS}"
        )
    return frame, {
        "predictions": str(path),
        "metadata": str(metadata_path),
        "score_columns": score_columns,
        "margin_columns": margin_columns,
    }


def _aligned_track(first: np.ndarray, second: np.ndarray, points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    if len(first) < 2 or len(second) < 2:
        return np.array([]), np.array([])
    grid = np.linspace(0.0, 1.0, points)
    first_values = np.interp(grid, np.linspace(0.0, 1.0, len(first)), first)
    second_values = np.interp(grid, np.linspace(0.0, 1.0, len(second)), second)
    return first_values, second_values


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3 or np.std(first[valid]) == 0 or np.std(second[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(first[valid], second[valid])[0, 1])


def objective_pair_metrics(
    original_path: Path,
    prosody_path: Path,
    sample_rate: int,
    frame_period_ms: float,
    f0_floor_hz: float,
    f0_ceiling_hz: float,
) -> dict[str, float]:
    kwargs = {
        "sample_rate": sample_rate,
        "frame_period_ms": frame_period_ms,
        "f0_floor_hz": f0_floor_hz,
        "f0_ceiling_hz": f0_ceiling_hz,
    }
    original = analyse_world(load_audio(original_path, sample_rate), **kwargs)
    prosody = analyse_world(load_audio(prosody_path, sample_rate), **kwargs)
    original_f0, prosody_f0 = _aligned_track(original.f0, prosody.f0)
    jointly_voiced = (original_f0 > 0) & (prosody_f0 > 0)
    log_original = np.log2(original_f0[jointly_voiced])
    log_prosody = np.log2(prosody_f0[jointly_voiced])
    original_energy, prosody_energy = _aligned_track(
        np.log(np.maximum(np.mean(original.spectral_envelope, axis=1), 1e-12)),
        np.log(np.maximum(np.mean(prosody.spectral_envelope, axis=1), 1e-12)),
    )
    return {
        "f0_correlation": _correlation(log_original, log_prosody),
        "energy_correlation": _correlation(original_energy, prosody_energy),
        "jointly_voiced_frames": int(jointly_voiced.sum()),
    }


def summarise_objective_metrics(
    detail: pd.DataFrame,
    f0_threshold: float,
    energy_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in detail.groupby("dataset", sort=True):
        row: dict[str, object] = {"dataset": dataset, "n": len(group)}
        for metric, threshold in (
            ("f0_correlation", f0_threshold),
            ("energy_correlation", energy_threshold),
        ):
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_pass_rate"] = float(values.ge(threshold).mean())
            row[f"{metric}_negative_rate"] = float(values.lt(0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paired O/P objective metrics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml"
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audio = config["audio"]
    thresholds = config["validation"]
    manifest = pd.read_csv(args.manifest, dtype={"id": str})
    require_columns(
        manifest, ("dataset", "id", "condition", "processed_audio_path"), args.manifest
    )
    wide = manifest.pivot(
        index=["dataset", "id"], columns="condition", values="processed_audio_path"
    ).reset_index()
    require_columns(wide, ("O", "P"), args.manifest)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for record in wide.to_dict(orient="records"):
        try:
            metrics = objective_pair_metrics(
                Path(record["O"]),
                Path(record["P"]),
                sample_rate=int(audio["sample_rate"]),
                frame_period_ms=float(audio["frame_period_ms"]),
                f0_floor_hz=float(audio["f0_floor_hz"]),
                f0_ceiling_hz=float(audio["f0_ceiling_hz"]),
            )
            rows.append({"dataset": record["dataset"], "id": record["id"], **metrics})
        except Exception as exc:
            errors.append(
                {"dataset": str(record["dataset"]), "id": str(record["id"]), "error": repr(exc)}
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("no O/P pairs were validated successfully")
    summary = summarise_objective_metrics(
        detail,
        f0_threshold=float(thresholds["f0_correlation_threshold"]),
        energy_threshold=float(thresholds["energy_correlation_threshold"]),
    )
    detail.to_csv(args.output_dir / "validation_objective_metrics_per_sample.csv", index=False)
    summary.to_csv(args.output_dir / "validation_objective_metrics.csv", index=False)
    (args.output_dir / "validation_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"valid pairs: {len(detail)}; errors: {len(errors)}")


if __name__ == "__main__":
    main()
