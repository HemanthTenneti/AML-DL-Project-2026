"""Phase 2A - DistilBERT end-to-end: [CLS] hypothesis [SEP] premise [SEP] -> valid / not valid.

No parser, no PC rules: the transformer must learn the reasoning from text alone.
It is the deep-learning counterpart of the TF-IDF + LR baseline.
"""
from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .dl_common import TrainConfig, encode, predict_proba, train_classifier

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_HEADER_VARS = re.compile(r"closed system of \d+ variables?, (.+?)\. All the statistical")
_SINGLE_CAPITAL = re.compile(r"\b[A-Z]\b")


def premise_variables(premise: str) -> list[str]:
    """Variables in the order the header lists them ('... 3 variables, A, B and C.')."""
    return _SINGLE_CAPITAL.findall(_HEADER_VARS.search(premise).group(1))


def rename_variables(premise: str, hypothesis: str, rng: random.Random) -> tuple[str, str]:
    """Renaming augmentation: give the variables new random letters, the same mapping in premise and
    hypothesis. Breaks the dataset shortcut 'alphabetical order = causal order'; the label is unchanged
    because renaming does not change any correlation or causal fact."""
    old = premise_variables(premise)
    mapping = dict(zip(old, rng.sample(LETTERS, len(old))))
    swap = lambda text: _SINGLE_CAPITAL.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
    return swap(premise), swap(hypothesis)


def sample_training_rows(train: pd.DataFrame, n_rows: int, tokenizer, max_length: int = 512,
                         seed: int = 42) -> pd.DataFrame:
    """Random subset of train, stratified by template x label, keeping only inputs that fit in
    `max_length` tokens (every test input fits; a cut-off premise could hide the facts the label depends on)."""
    pool = train.sample(min(len(train), 3 * n_rows), random_state=seed)
    lengths = np.array([len(r) for r in encode(tokenizer, pool.hypothesis, pool.premise, max_length=10_000)])
    fits = pool[lengths <= max_length]
    frac = min(1.0, n_rows / len(fits))
    return (fits.groupby(["template", "label"], group_keys=False)
                .sample(frac=frac, random_state=seed)
                .sample(frac=1.0, random_state=seed))  # shuffle


def augment(rows: pd.DataFrame, p_rename: float, seed: int = 42) -> pd.DataFrame:
    """Rename the variables of a random `p_rename` share of the rows (static augmentation, reproducible)."""
    rng = random.Random(seed)
    premises, hypotheses = [], []
    for premise, hypothesis in zip(rows.premise, rows.hypothesis):
        if rng.random() < p_rename:
            premise, hypothesis = rename_variables(premise, hypothesis, rng)
        premises.append(premise)
        hypotheses.append(hypothesis)
    return rows.assign(premise=premises, hypothesis=hypotheses)


def valid_probability(model, tokenizer, df: pd.DataFrame, max_length: int = 512) -> np.ndarray:
    """P(valid) for every row of a Corr2Cause frame (columns premise, hypothesis)."""
    return predict_proba(model, tokenizer, df.hypothesis, df.premise, max_length=max_length)[:, 1]


def train_end2end(rows: pd.DataFrame, dev: pd.DataFrame, config: TrainConfig, log=print):
    """Fine-tune on (hypothesis, premise) pairs; keep the epoch with the best dev F1."""
    def dev_f1(model, tokenizer) -> float:
        prob = valid_probability(model, tokenizer, dev, config.max_length)
        return f1_score(dev.label, prob >= 0.5, zero_division=0)

    return train_classifier(rows.hypothesis, rows.premise, rows.label, num_labels=2,
                            config=config, dev_score=dev_f1, log=log)
