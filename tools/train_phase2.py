"""Train the three Phase 2 models and save them under models/ (the notebook loads them).

    .venv/bin/python tools/train_phase2.py            # all models (~1 h on an Apple M3 Pro)
    .venv/bin/python tools/train_phase2.py parser     # only the 2B parser (~2 min)

models/parser/            2B  DistilBERT hypothesis parser (relation + direction)
models/end2end/           2A  DistilBERT end-to-end classifier
models/end2end_renamed/   2A  same, with variable-renaming augmentation
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from causal_nlp import dl_end2end, dl_parser  # noqa: E402
from causal_nlp.data import load_split  # noqa: E402
from causal_nlp.dl_common import MODEL_NAME, MODELS_DIR, TrainConfig, predict_proba, save, train_classifier  # noqa: E402

# Laptop budget (Apple M3 Pro, 18 GB shared memory: batch 16 x 512 tokens swaps). On a bigger GPU,
# e.g. Colab:  END2END_ROWS=40000 END2END_BATCH=16 python tools/train_phase2.py end2end
END2END_ROWS = int(os.environ.get("END2END_ROWS", 12_000))
END2END_CONFIG = TrainConfig(epochs=2, batch_size=int(os.environ.get("END2END_BATCH", 8)), lr=2e-5)
PARSER_CONFIG = TrainConfig(epochs=3, batch_size=32, lr=5e-5, max_length=64)
PARSER_SENTENCES = {"original": 1500, "grammar": 3000, "validation": 500}  # per relation


def log_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")

    def log(message: str) -> None:
        print(message, flush=True)
        handle.write(message + "\n")
        handle.flush()
    return log


def train_parser() -> None:
    train_templates, val_templates = dl_parser.training_templates()
    train = pd.concat([
        dl_parser.make_examples(dl_parser.ORIGINAL_TEMPLATES, PARSER_SENTENCES["original"], seed=1),
        dl_parser.make_examples(train_templates, PARSER_SENTENCES["grammar"], seed=2),
    ]).sample(frac=1.0, random_state=0)
    val = dl_parser.make_examples(val_templates, PARSER_SENTENCES["validation"], seed=3)
    log = log_to(MODELS_DIR / "parser.log")
    log(f"parser: {len(train)} training sentences, {len(val)} validation sentences (held-out verbs)")

    def val_accuracy(model, tokenizer) -> float:
        pred = predict_proba(model, tokenizer, val.text, None, max_length=PARSER_CONFIG.max_length).argmax(1)
        return float((pred == val.label.values).mean())

    model, tokenizer, history = train_classifier(train.text, None, train.label, len(dl_parser.LABELS),
                                                 PARSER_CONFIG, dev_score=val_accuracy, log=log)
    save(model, tokenizer, history, PARSER_CONFIG, MODELS_DIR / "parser",
         extra={"labels": dl_parser.LABELS, "n_train": len(train), "n_val": len(val)})


def train_end2end(renamed: bool) -> None:
    name = "end2end_renamed" if renamed else "end2end"
    log = log_to(MODELS_DIR / f"{name}.log")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    rows = dl_end2end.sample_training_rows(load_split("train"), END2END_ROWS, tokenizer, seed=END2END_CONFIG.seed)
    if renamed:
        rows = dl_end2end.augment(rows, p_rename=0.5, seed=END2END_CONFIG.seed)
    log(f"{name}: {len(rows)} training rows, {rows.label.mean():.1%} valid, "
        f"{rows.premise.nunique()} premises")
    model, tokenizer, history = dl_end2end.train_end2end(rows, load_split("dev"), END2END_CONFIG, log=log)
    save(model, tokenizer, history, END2END_CONFIG, MODELS_DIR / name,
         extra={"n_train": len(rows), "renaming_augmentation": renamed})


if __name__ == "__main__":
    targets = sys.argv[1:] or ["parser", "end2end", "end2end_renamed"]
    if "parser" in targets:
        train_parser()
    if "end2end" in targets:
        train_end2end(renamed=False)
    if "end2end_renamed" in targets:
        train_end2end(renamed=True)
