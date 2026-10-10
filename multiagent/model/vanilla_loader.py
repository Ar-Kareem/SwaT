from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from multiagent.paths import MODELS_ROOT


MODEL_PATHS = {
    "llama2-13b-chat": "llama/Llama-2-13b-chat-hf",
    "gpt-neo-2.7b": "eleutherai/gpt-neo-2.7B",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-4-e2b-it": "google/gemma-4-E2B-it",
    "gemma-4-e4b-it": "google/gemma-4-E4B-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "llama3-8b": "llama3/models--meta-llama--Meta-Llama-3-8B/snapshots/1460c22666392e470910ce3d44ffeb2ab7dbd4df",
    "llama3-8b-instruct": "llama3/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45",
    "llama3.1-8b-instruct": "llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.2-11b-vision-instruct": "llama3/hf/11B-Vision-Instruct",
    "granite-3.3-8b-instruct": "ibm-granite/granite-3.3-8b-instruct",
    "ministral-3-3b-instruct": "mistralai/Ministral-3-3B-Instruct-2512",
    "ministral-3-8b-instruct": "mistralai/Ministral-3-8B-Instruct-2512",
    "ministral-3-14b-instruct": "mistralai/Ministral-3-14B-Instruct-2512",
    "phi-2": "microsoft/phi-2",
    "pythia-2.8b": "eleutherai/pythia-2.8b",
    "qwen3-0.6b": "qwen/Qwen3-0.6B",
    "qwen3-4b": "qwen/Qwen3-4B",
    "qwen3-4b-2507": "qwen/Qwen3-4B-Instruct-2507",
    "qwen3-8b": "qwen/Qwen3-8B",
    "qwen3-14b": "qwen/Qwen3-14B",
    "qwen3-32b": "qwen/Qwen3-32B",
    "qwen3-32b-fp8": "qwen/Qwen3-32B-FP8",
    "qwen3-4b-base": "qwen3/Qwen3-4B-Base",
    "qwen3.5-0.8b": "qwen/Qwen3.5-0.8B",
    "qwen3.5-2b": "qwen/Qwen3.5-2B",
    "qwen3.5-4b": "qwen/Qwen3.5-4B",
    "qwen3.5-9b": "qwen/Qwen3.5-9B",
    "qwen3.5-27b": "qwen/Qwen3.5-27B",
    "qwen3.5-35b-a3b": "qwen/Qwen3.5-35B-A3B",
    "tinyllama-1.1b-chat-v1.0": "tinyllama/TinyLlama-1.1B-Chat-v1.0",
}

MULTIMODAL_MODEL_TYPES = {"qwen3_5"}
IMAGE_TEXT_MODEL_TYPES = {"mistral3", "mllama"}


def causal_model_class(config):
    if config.model_type in MULTIMODAL_MODEL_TYPES:
        return transformers.AutoModelForMultimodalLM
    if config.model_type in IMAGE_TEXT_MODEL_TYPES:
        return transformers.AutoModelForImageTextToText
    return AutoModelForCausalLM


def from_pretrained_kwargs(config) -> dict:
    qc = getattr(config, "quantization_config", None)
    method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    if method == "fp8":
        return {"quantization_config": transformers.FineGrainedFP8Config(dequantize=True)}
    return {}


def load_causal_model(path, dtype, device: str):
    config = AutoConfig.from_pretrained(path)
    model = causal_model_class(config).from_pretrained(path, dtype=dtype, **from_pretrained_kwargs(config))
    return model.to(device).eval()


@dataclass
class LoadedModel:
    name: str
    path: Path
    tokenizer: object
    model: object
    device: str


def resolve_model_path(name: str, root: str | Path = MODELS_ROOT) -> Path:
    path = Path(name)
    if path.exists():
        return path

    root = Path(root)
    normalized_name = name.lower()
    if normalized_name in MODEL_PATHS:
        return root / MODEL_PATHS[normalized_name]

    path = root / name
    if path.exists():
        return path

    config_paths = list(root.rglob("config.json"))
    matches = [p for p in config_paths if p.parent.name.lower() == normalized_name]
    if len(matches) == 1:
        return matches[0].parent
    if matches:
        raise ValueError(f"ambiguous model name {name!r}: {[str(p.parent) for p in matches]}")

    matches = [p for p in config_paths if normalized_name in str(p.parent).lower()]
    if len(matches) == 1:
        return matches[0].parent
    if matches:
        raise ValueError(f"ambiguous model name {name!r}: {[str(p.parent) for p in matches]}")
    raise FileNotFoundError(f"model not found under {root}: {name}")


def load_model(name: str, device: str | None = None, root: str | Path = MODELS_ROOT) -> LoadedModel:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve_model_path(name, root)
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    dtype = torch.float32 if device == "cpu" else dtype

    config = AutoConfig.from_pretrained(path)
    model_cls = causal_model_class(config)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    else:
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model = model_cls.from_pretrained(path, dtype=dtype, **from_pretrained_kwargs(config)).to(device).eval()
    return LoadedModel(name=name, path=path, tokenizer=tokenizer, model=model, device=device)
