"""Phase 2B - DistilBERT as the language layer: hypothesis sentence -> Hypothesis(relation, x, y).

Drop-in replacement for `parsing.parse_hypothesis`, so the PC rules, features, decision tree and
graph stage stay exactly the same as in Phase 1.

How it reads a sentence:
  * the variables are the single capital letters, in order of appearance (same as the regex parser);
  * DistilBERT only decides the relation *type* and its *direction*:
        "B is directly caused by A"  ->  direct_reversed  ->  Hypothesis(direct, x=A, y=B)

Training data = the original Corr2Cause templates + sentences from our own small grammar
(`paraphrase_grammar.py`), filled with random letters. The official paraphrase test wordings are never used for training
(`check_no_test_leak` enforces it).
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .dl_common import load, predict_proba
from .paraphrase_grammar import all_templates
from .parsing import COLLIDER, CONFOUNDER, DIRECT, INDIRECT, Hypothesis

LABELS = ("direct", "direct_reversed", "indirect", "indirect_reversed", "collider", "confounder")
LABEL_ID = {name: i for i, name in enumerate(LABELS)}
SYMMETRIC = (COLLIDER, CONFOUNDER)
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# The exact templates of the Corr2Cause train split ({x} = cause slot, {y} = effect slot).
ORIGINAL_TEMPLATES = {
    DIRECT: ["{x} directly causes {y}."],
    INDIRECT: ["{x} causes something else which causes {y}.", "{x} is a cause for {y}, but not a direct one."],
    COLLIDER: ["There exists at least one collider (i.e., common effect) of {x} and {y}."],
    CONFOUNDER: ["There exists at least one confounder (i.e., common cause) of {x} and {y}."],
}

# Validation = every grammar sentence that uses one of these verbs (never seen in training).
HELD_OUT_VERBS = ("triggers", "trigger", "triggered", "generates", "generate", "generated")

# Key words of the 4 official paraphrase test wordings (Jin et al.) - must never appear in training:
#   "A directly affects B." / "A influences B through some mediator(s)." /
#   "A and B together cause some other variable(s)." / "Some variable(s) cause(s) both A and B."
TEST_KEY_PHRASES = ("affects", "mediator", "together cause", "some other variable", "cause(s) both")
_VARIABLE = re.compile(r"\b[A-Z]\b")


def template_label(template: str, relation: str) -> str:
    """Relation + direction as seen in the sentence: is the cause mentioned first or second?"""
    if relation in SYMMETRIC:
        return relation
    return relation if template.index("{x}") < template.index("{y}") else f"{relation}_reversed"


def check_no_test_leak(templates: dict[str, list[str]]) -> None:
    """Raise if any training template uses a key phrase of the paraphrase test set."""
    for items in templates.values():
        for template in items:
            if any(k in template.lower() for k in TEST_KEY_PHRASES):
                raise ValueError(f"training template overlaps the paraphrase test set: {template!r}")


def make_examples(templates: dict[str, list[str]], n_per_relation: int, seed: int = 42) -> pd.DataFrame:
    """Fill templates with random distinct letters -> (text, label, relation, x, y, template).
    Every relation gets about `n_per_relation` sentences, spread evenly over its templates."""
    rng = random.Random(seed)
    rows = []
    for relation, items in templates.items():
        per_template = max(1, round(n_per_relation / len(items)))
        for template in items:
            label = template_label(template, relation)
            for _ in range(per_template):
                x, y = rng.sample(LETTERS, 2)
                rows.append({"text": template.format(x=x, y=y), "label": LABEL_ID[label],
                             "relation": relation, "x": x, "y": y, "template": template})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def training_templates() -> tuple[dict, dict]:
    """(train = grammar without the held-out verbs, validation = grammar sentences with them).
    The original Corr2Cause templates are returned separately by `ORIGINAL_TEMPLATES`."""
    grammar = all_templates()
    check_no_test_leak(grammar)
    uses_held_out = lambda t: any(re.search(rf"\b{v}\b", t) for v in HELD_OUT_VERBS)
    train = {rel: [t for t in items if not uses_held_out(t)] for rel, items in grammar.items()}
    val = {rel: [t for t in items if uses_held_out(t)] for rel, items in grammar.items()}
    return train, val


def variables_in_order(text: str) -> list[str]:
    seen = []
    for v in _VARIABLE.findall(text):
        if v not in seen:
            seen.append(v)
    return seen


def decode(label: str, text: str) -> Hypothesis | None:
    """Predicted label + the letters in the sentence -> Hypothesis (None if not exactly two variables)."""
    variables = variables_in_order(text)
    if len(variables) != 2:
        return None
    first, second = variables
    if label.endswith("_reversed"):
        return Hypothesis(label.removesuffix("_reversed"), second, first)
    return Hypothesis(label, first, second)


class DistilBertParser:
    """Callable like `parse_hypothesis`: text -> Hypothesis | None. Call `parse_many` first to batch the GPU work."""

    def __init__(self, model_dir: Path):
        self.model, self.tokenizer, _ = load(model_dir)
        self._cache: dict[str, Hypothesis | None] = {}  # each distinct sentence goes through DistilBERT once

    def parse_many(self, texts) -> list[Hypothesis | None]:
        todo = sorted({t.strip() for t in texts} - self._cache.keys())
        if todo:
            probs = predict_proba(self.model, self.tokenizer, todo, None, max_length=64)
            for text, p in zip(todo, probs):
                self._cache[text] = decode(LABELS[int(np.argmax(p))], text)
        return [self._cache[t.strip()] for t in texts]

    def __call__(self, text: str) -> Hypothesis | None:
        return self.parse_many([text])[0]
