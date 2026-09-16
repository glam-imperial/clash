from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from preprocessing.world_processing import (
    WorldFeatures,
    analyse_world,
    estimate_carrier,
    frame_power,
    load_audio,
    save_audio,
    scale_to_frame_power,
    synthesise_world,
)


@dataclass(frozen=True)
class NeutralisationParameters:
    target_f0_median_hz: float
    target_f0_spread_semitones: float


def neutralise_f0(
    f0: np.ndarray,
    target_median_hz: float,
    target_spread_semitones: float,
) -> np.ndarray:
    """Compress utterance-level pitch variation while preserving voicing."""
    result = np.asarray(f0, dtype=np.float64).copy()
    voiced = result > 0
    if not voiced.any():
        return result
    semitones = 12.0 * np.log2(result[voiced] / np.median(result[voiced]))
    spread = float(np.std(semitones))
    if target_spread_semitones <= 0 or spread <= 1e-8:
        semitones = np.zeros_like(semitones)
    else:
        semitones *= target_spread_semitones / spread
    result[voiced] = target_median_hz * np.power(2.0, semitones / 12.0)
    return result


def lexical_only(
    original: WorldFeatures,
    parameters: NeutralisationParameters,
) -> WorldFeatures:
    constant_power = np.full_like(
        original.f0,
        float(np.median(frame_power(original.spectral_envelope))),
        dtype=np.float64,
    )
    return original.updated(
        f0=neutralise_f0(
            original.f0,
            parameters.target_f0_median_hz,
            parameters.target_f0_spread_semitones,
        ),
        spectral_envelope=scale_to_frame_power(
            original.spectral_envelope, constant_power
        ),
    )


def prosody_only(original: WorldFeatures, carrier: np.ndarray) -> WorldFeatures:
    carrier_frames = np.tile(np.asarray(carrier, dtype=np.float64), (len(original.f0), 1))
    return original.updated(
        spectral_envelope=scale_to_frame_power(
            carrier_frames, frame_power(original.spectral_envelope)
        )
    )


def fully_neutralised(
    original: WorldFeatures,
    carrier: np.ndarray,
    parameters: NeutralisationParameters,
) -> WorldFeatures:
    carrier_frames = np.tile(np.asarray(carrier, dtype=np.float64), (len(original.f0), 1))
    constant_power = np.full_like(
        original.f0,
        float(np.median(frame_power(original.spectral_envelope))),
        dtype=np.float64,
    )
    fixed_aperiodicity = np.tile(
        np.median(original.aperiodicity, axis=0), (len(original.f0), 1)
    )
    return original.updated(
        f0=neutralise_f0(
            original.f0,
            parameters.target_f0_median_hz,
            parameters.target_f0_spread_semitones,
        ),
        spectral_envelope=scale_to_frame_power(carrier_frames, constant_power),
        aperiodicity=fixed_aperiodicity,
    )


def build_conditions(
    original: WorldFeatures,
    carrier: np.ndarray,
    parameters: NeutralisationParameters,
) -> dict[str, WorldFeatures]:
    return {
        "O": original,
        "L": lexical_only(original, parameters),
        "P": prosody_only(original, carrier),
        "F": fully_neutralised(original, carrier, parameters),
    }


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset", "id", "audio_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain {sorted(required)}")
    return rows


def _audio_config(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config["audio"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct paired O/L/P/F audio controls.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml"
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_manifest(args.manifest)
    audio = _audio_config(args.config)
    sample_rate = int(audio["sample_rate"])
    analysis_args = {
        "sample_rate": sample_rate,
        "frame_period_ms": float(audio["frame_period_ms"]),
        "f0_floor_hz": float(audio["f0_floor_hz"]),
        "f0_ceiling_hz": float(audio["f0_ceiling_hz"]),
    }
    cache: dict[tuple[str, str], WorldFeatures] = {}
    by_dataset: dict[str, list[WorldFeatures]] = {}
    for row in rows:
        features = analyse_world(load_audio(Path(row["audio_path"]), sample_rate), **analysis_args)
        cache[(row["dataset"], row["id"])] = features
        by_dataset.setdefault(row["dataset"], []).append(features)
    carriers = {name: estimate_carrier(items) for name, items in by_dataset.items()}

    output_rows: list[dict[str, str]] = []
    spreads = audio["lexical_f0_spread_semitones"]
    for row in rows:
        dataset = row["dataset"]
        params = NeutralisationParameters(
            target_f0_median_hz=float(audio["target_f0_median_hz"]),
            target_f0_spread_semitones=float(spreads[dataset]),
        )
        controls = build_conditions(
            cache[(dataset, row["id"])], carriers[dataset], params
        )
        for condition, features in controls.items():
            destination = args.output_dir / dataset / condition / f"{row['id']}.wav"
            waveform = (
                load_audio(Path(row["audio_path"]), sample_rate)
                if condition == "O"
                else synthesise_world(features)
            )
            save_audio(destination, waveform, sample_rate)
            output_rows.append(
                {**row, "condition": condition, "processed_audio_path": str(destination)}
            )
    output_manifest = args.output_dir / "conditions_manifest.csv"
    with output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} rows to {output_manifest}")


if __name__ == "__main__":
    main()
