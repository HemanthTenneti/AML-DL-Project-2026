"""Shared DistilBERT helpers for Phase 2: device, tokenisation, a plain training loop, prediction.

A hand-written PyTorch loop (instead of the Hugging Face `Trainer`) so every step can be
explained: tokenise -> batch -> forward -> weighted cross-entropy -> backward -> AdamW step.
"""
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "distilbert-base-cased"  # cased: variables are capital letters ("A" must not become the word "a")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PAD_MULTIPLE = 64  # pad batch widths to 64, 128, ... so the GPU sees few distinct shapes (MPS caches one kernel per shape)


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = MODEL_NAME
    max_length: int = 512
    epochs: int = 2
    batch_size: int = 16
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(tokenizer, first: Sequence[str], second: Sequence[str] | None, max_length: int) -> list[list[int]]:
    """Token ids per example. For pairs, only the second text (the long premise) is truncated."""
    if second is None:
        enc = tokenizer(list(first), truncation=True, max_length=max_length)
    else:
        enc = tokenizer(list(first), list(second), truncation="only_second", max_length=max_length)
    return enc["input_ids"]


def _pad(ids: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = -(-max(map(len, ids)) // PAD_MULTIPLE) * PAD_MULTIPLE
    input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), width), dtype=torch.long)
    for i, row in enumerate(ids):
        input_ids[i, :len(row)] = torch.tensor(row)
        mask[i, :len(row)] = 1
    return input_ids, mask


def length_grouped_batches(lengths: Sequence[int], batch_size: int, rng: random.Random) -> list[list[int]]:
    """Shuffle, then sort inside chunks of 50 batches so each batch holds similar lengths (less padding)."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    chunk = batch_size * 50
    batches = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start:start + chunk], key=lambda i: lengths[i])
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def predict_proba(model, tokenizer, first: Sequence[str], second: Sequence[str] | None = None,
                  max_length: int = 512, batch_size: int = 64) -> np.ndarray:
    """Class probabilities, shape (n_examples, n_classes). Sorted by length for speed, returned in input order."""
    model.eval()
    device = next(model.parameters()).device
    ids = encode(tokenizer, first, second, max_length)
    order = np.argsort([len(r) for r in ids])
    probs = np.zeros((len(ids), model.config.num_labels), dtype=np.float32)
    for start in range(0, len(ids), batch_size):
        idx = order[start:start + batch_size]
        input_ids, mask = _pad([ids[i] for i in idx], tokenizer.pad_token_id)
        logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


def train_classifier(first: Sequence[str], second: Sequence[str] | None, labels: Sequence[int], num_labels: int,
                     config: TrainConfig, dev_score: Callable[[object, object], float] | None = None,
                     log: Callable[[str], None] = print):
    """Fine-tune `config.model_name` for sequence classification.

    dev_score(model, tokenizer) -> float is called after every epoch; the best epoch is kept
    (early stopping). Returns (model, tokenizer, history)."""
    set_seed(config.seed)
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=num_labels).to(device)

    ids = encode(tokenizer, first, second, config.max_length)
    y = torch.tensor(list(labels), dtype=torch.long)
    counts = torch.bincount(y, minlength=num_labels).float()
    weights = counts.sum() / (num_labels * counts.clamp(min=1))  # class weights: only ~21 % of claims are valid
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))

    rng = random.Random(config.seed)
    steps_per_epoch = -(-len(ids) // config.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.warmup_ratio * steps_per_epoch * config.epochs),
                                                steps_per_epoch * config.epochs)
    history, best_score, best_state = [], -1.0, None
    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        batches = length_grouped_batches([len(r) for r in ids], config.batch_size, rng)
        for step, batch in enumerate(batches, 1):
            input_ids, mask = _pad([ids[i] for i in batch], tokenizer.pad_token_id)
            logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits, y[batch].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if device.type == "mps" and step % 100 == 0:
                torch.mps.empty_cache()
            if step % 200 == 0 or step == len(batches):
                log(f"epoch {epoch} step {step}/{len(batches)} loss {running / step:.4f} ({time.time() - t0:.0f}s)")
        record = {"epoch": epoch, "train_loss": running / len(batches), "seconds": time.time() - t0}
        if dev_score is not None:
            record["dev_score"] = dev_score(model, tokenizer)
            log(f"epoch {epoch} dev score {record['dev_score']:.4f}")
            if record["dev_score"] > best_score:
                best_score = record["dev_score"]
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        history.append(record)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer, history


def save(model, tokenizer, history: list[dict], config: TrainConfig, folder: Path, extra: dict | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    info = {"config": asdict(config), "history": history,
            "n_parameters": sum(p.numel() for p in model.parameters()), **(extra or {})}
    (folder / "training_info.json").write_text(json.dumps(info, indent=2))


def load(folder: Path, device: torch.device | None = None):
    """-> (model, tokenizer, training_info)"""
    device = device or pick_device()
    model = AutoModelForSequenceClassification.from_pretrained(folder).to(device)
    tokenizer = AutoTokenizer.from_pretrained(folder)
    info = json.loads((folder / "training_info.json").read_text())
    return model, tokenizer, info
