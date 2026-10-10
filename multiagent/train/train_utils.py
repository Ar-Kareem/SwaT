from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from multiagent.model.unaware_comm import cache_layer, full_sender_cache
from multiagent.model.vanilla_loader import load_causal_model
from multiagent.train.loss_plot import write_html


def load_model(path: Path, dtype: torch.dtype, device: str):
    model = load_causal_model(path, dtype, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer


def eos_ids(tokenizer) -> set[int]:
    value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, list):
        return {int(item) for item in value}
    return {int(value)}


def move_kv_cache(cache, device: str):
    if hasattr(cache, "key_cache"):
        layer_count = len(cache.key_cache)
    elif hasattr(cache, "layers"):
        layer_count = len(cache.layers)
    else:
        layer_count = len(cache)
    pairs = (cache_layer(cache, index) for index in range(layer_count))
    return tuple((key.to(device), value.to(device)) for key, value in pairs if key is not None and key.numel() > 0)


@torch.no_grad()
def prefill_all(sender_model, receiver_model, batch):
    sender_inputs = {**batch["sender"], "use_cache": True, "past_key_values": full_sender_cache()}
    receiver_inputs = {**batch["receiver"], "use_cache": True}

    def sender_task():
        return sender_model(**sender_inputs).past_key_values

    def receiver_task():
        return receiver_model(**receiver_inputs).past_key_values

    sender_device = sender_inputs["input_ids"].device
    receiver_device = receiver_inputs["input_ids"].device
    if sender_device.type == "cuda" and receiver_device.type == "cuda" and sender_device != receiver_device:
        with ThreadPoolExecutor(max_workers=2) as executor:
            sender_future = executor.submit(sender_task)
            receiver_future = executor.submit(receiver_task)
            return sender_future.result(), receiver_future.result()
    return sender_task(), receiver_task()


def metric_from_grade(result) -> float:
    if isinstance(result, dict):
        if "answer_f1" in result:
            return float(result["answer_f1"])
        result = result.get("grade")
    return 1.0 if str(result) == "correct" else 0.0


def save_json(path: Path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def save_losses_and_plot(output_dir: Path, losses: list[dict]):
    losses_path = output_dir / "losses.json"
    save_json(losses_path, losses)
    if losses:
        write_html(losses_path, output_dir / "losses.html")


def cycler(loader):
    while True:
        yield from loader
