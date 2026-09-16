from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from preprocessing.validation import require_columns


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def logsumexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(array))
    return maximum + math.log(float(np.exp(array - maximum).sum()))


def token_set_margin(
    logits: np.ndarray,
    yes_token_ids: list[int],
    no_token_ids: list[int],
) -> tuple[float, float, float]:
    values = np.asarray(logits, dtype=np.float64)
    yes = logsumexp(values[yes_token_ids])
    no = logsumexp(values[no_token_ids])
    return yes - no, yes, no


def single_token_margin(
    logits: np.ndarray, yes_token_id: int, no_token_id: int
) -> float:
    values = np.asarray(logits, dtype=np.float64)
    return float(values[yes_token_id] - values[no_token_id])


def resolve_single_token_variants(tokenizer: Any, variants: list[str]) -> list[int]:
    token_ids: list[int] = []
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1:
            token_ids.append(int(encoded[0]))
    token_ids = list(dict.fromkeys(token_ids))
    if not token_ids:
        raise ValueError(
            "none of the configured answer variants maps to exactly one token; "
            "update configs/prompt.yaml for this tokenizer"
        )
    return token_ids


def build_messages(
    audio_path: Path,
    setting: str,
    context: str,
    prompts: dict[str, Any],
) -> list[dict[str, Any]]:
    if setting == "target_only":
        question = str(prompts["target_only"])
    elif setting == "audio_plus_context":
        question = str(prompts["audio_plus_context"]).format(context=context)
    else:
        raise ValueError(f"unsupported setting: {setting}")
    return [
        {"role": "system", "content": [{"type": "text", "text": prompts["system"]}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path.resolve())},
                {"type": "text", "text": question},
            ],
        },
    ]


def load_model(model_name: str, config: dict[str, Any]):
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("inference requires torch and transformers") from exc
    entry = config["models"][model_name]
    model_path = str(entry.get("path_or_id", "")).strip()
    processor_path = str(entry.get("processor_path_or_id", "")).strip() or model_path
    class_name = str(entry.get("transformers_class", "")).strip()
    if not model_path or not class_name:
        raise ValueError(
            f"set models.{model_name}.path_or_id and transformers_class in model.yaml"
        )
    model_class = getattr(transformers, class_name, None)
    if model_class is None:
        raise ValueError(f"transformers has no class named {class_name}")
    inference = config["inference"]
    dtype_name = str(inference.get("dtype", "bfloat16"))
    dtype = getattr(torch, dtype_name)
    processor = transformers.AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=bool(inference.get("trust_remote_code", True)),
    )
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=inference.get("device_map", "auto"),
        trust_remote_code=bool(inference.get("trust_remote_code", True)),
    )
    model.eval()
    return model, processor


def decision_logits(model: Any, processor: Any, messages: list[dict[str, Any]]) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("inference requires torch") from exc
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model(**inputs, use_cache=False)
    logits = output.logits[0, -1].detach().float().cpu().numpy()
    if logits.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError("model returned invalid first-decision logits")
    return logits


def run_inference(
    manifest_path: Path,
    output_path: Path,
    setting: str,
    model_name: str,
    model_config_path: Path,
    prompt_config_path: Path,
) -> pd.DataFrame:
    model_config = load_yaml(model_config_path)
    prompts = load_yaml(prompt_config_path)
    set_deterministic_seed(int(model_config["inference"]["seed"]))
    manifest = pd.read_csv(manifest_path, dtype={"id": str})
    require_columns(manifest, ("dataset", "id", "condition", "audio_path"), manifest_path)
    if setting == "audio_plus_context":
        require_columns(manifest, ("context_text",), manifest_path)
    model, processor = load_model(model_name, model_config)
    tokenizer = getattr(processor, "tokenizer", processor)
    yes_ids = resolve_single_token_variants(tokenizer, list(prompts["choices"]["yes"]))
    no_ids = resolve_single_token_variants(tokenizer, list(prompts["choices"]["no"]))

    rows: list[dict[str, Any]] = []
    for record in manifest.to_dict(orient="records"):
        result = dict(record)
        result.update(
            {
                "model": model_name,
                "setting": setting,
                "score_method": "token_set_logsumexp",
                "score": np.nan,
                "token_margin": np.nan,
                "single_token_margin": np.nan,
                "yes_logsumexp": np.nan,
                "no_logsumexp": np.nan,
                "error": "",
            }
        )
        try:
            logits = decision_logits(
                model,
                processor,
                build_messages(
                    Path(str(record["audio_path"])),
                    setting,
                    str(record.get("context_text", "")),
                    prompts,
                ),
            )
            margin, yes_score, no_score = token_set_margin(logits, yes_ids, no_ids)
            result.update(
                {
                    "score": margin,
                    "token_margin": margin,
                    "single_token_margin": single_token_margin(logits, yes_ids[0], no_ids[0]),
                    "yes_logsumexp": yes_score,
                    "no_logsumexp": no_score,
                }
            )
        except Exception as exc:
            result["error"] = repr(exc)
        rows.append(result)
    output = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run audio-LLM scoring at the first answer position."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--setting", choices=("target_only", "audio_plus_context"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-config", type=Path, default=REPOSITORY_ROOT / "configs" / "model.yaml"
    )
    parser.add_argument(
        "--prompt-config", type=Path, default=REPOSITORY_ROOT / "configs" / "prompt.yaml"
    )
    args = parser.parse_args()
    output = run_inference(
        args.manifest,
        args.output,
        args.setting,
        args.model,
        args.model_config,
        args.prompt_config,
    )
    errors = int(output["error"].astype(bool).sum())
    print(f"rows: {len(output)}; errors: {errors}; output: {args.output}")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

