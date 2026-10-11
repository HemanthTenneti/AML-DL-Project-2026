"""Loading Corr2Cause splits.

Labels come from `zluke/corr2cause-corrected` (same text as the official
`causal-nlp/corr2cause`, 6,943 labels fixed after a bug in the original generator).
The official split is already grouped by premise (no premise appears in two splits),
so there is no text leakage between train, dev and test.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from .parsing import COLLIDER, CONFOUNDER, DIRECT, INDIRECT, parse_hypothesis, split_input

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CORRECTED_REPO, OFFICIAL_REPO = "zluke/corr2cause-corrected", "causal-nlp/corr2cause"

# Paraphrased hypothesis templates used in the official robustness test set.
PARAPHRASE_PATTERNS = [
    (re.compile(r"^([A-Z]) directly affects ([A-Z])\.$"), DIRECT),
    (re.compile(r"^([A-Z]) influences ([A-Z]) through some mediator\(s\)\.$"), INDIRECT),
    (re.compile(r"^([A-Z]) and ([A-Z]) together cause some other variable\(s\)\.$"), COLLIDER),
    (re.compile(r"^Some variable\(s\) cause\(s\) both ([A-Z]) and ([A-Z])\.$"), CONFOUNDER),
]


def _download(repo: str, filename: str, folder: str) -> Path:
    target = DATA_DIR / folder / filename
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(hf_hub_download(repo, filename, repo_type="dataset")).read_bytes())
    return target


def _with_columns(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["input"].map(split_input)
    premise = parts.str[0]
    return df.assign(premise=premise, hypothesis=parts.str[1],
                     premise_id=pd.factorize(premise)[0])


def load_split(split: str, corrected: bool = True) -> pd.DataFrame:
    """split in {'train', 'dev', 'test'}."""
    repo, folder = (CORRECTED_REPO, "corrected") if corrected else (OFFICIAL_REPO, "raw")
    df = pd.read_csv(_download(repo, f"{split}.csv", folder))
    return _with_columns(df[["input", "label", "num_variables", "template"]])


def _key(premise: str, relation: str, x: str, y: str) -> tuple:
    """Collider / confounder claims are symmetric: 'of A and B' == 'of B and A'."""
    if relation in (COLLIDER, CONFOUNDER):
        x, y = sorted((x, y))
    return premise, relation, x, y


def _corrected_lookup() -> dict:
    """(premise, relation, x, y) -> corrected label, for test and dev."""
    lookup = {}
    for split in ("test", "dev"):
        df = load_split(split)
        for premise, hyp_text, label in zip(df.premise, df.hypothesis, df.label):
            h = parse_hypothesis(hyp_text)
            lookup[_key(premise, h.relation, h.x, h.y)] = label
    return lookup


def _rename_to_abc(premise: str, hyp: str) -> tuple[str, str, dict]:
    """Undo variable refactoring: the header lists variables in order, map them to A, B, C..."""
    header = re.search(r"variables?, (.+?)\. All the statistical", premise).group(1)
    names = re.findall(r"\b[A-Z]\b", header)
    mapping = {old: chr(ord("A") + i) for i, old in enumerate(names)}
    swap = lambda text: re.sub(r"\b[A-Z]\b", lambda m: mapping.get(m.group(0), m.group(0)), text)
    return swap(premise), swap(hyp), mapping


def load_refactorization_test() -> pd.DataFrame:
    """Same reasoning problems, variables renamed (e.g. A,B,C -> Z,Y,X)."""
    df = pd.read_csv(_download(OFFICIAL_REPO, "perturbation_by_refactorization_test.csv", "raw"))
    df = _with_columns(df[["input", "label"]])
    lookup = _corrected_lookup()
    fixed = []
    for premise, hyp_text, label in zip(df.premise, df.hypothesis, df.label):
        p, h, _ = _rename_to_abc(premise, hyp_text)
        hyp = parse_hypothesis(h)
        fixed.append(lookup.get(_key(p, hyp.relation, hyp.x, hyp.y), label) if hyp else label)
    return df.assign(label=fixed)


def load_paraphrase_test() -> pd.DataFrame:
    """Same premises as test, hypotheses re-worded (e.g. 'directly causes' -> 'directly affects')."""
    df = pd.read_csv(_download(OFFICIAL_REPO, "perturbation_by_paraphrasing_test.csv", "raw"))
    df["input"] = "Premise: " + df.premise.str.strip() + "\nHypothesis: " + df.hypothesis.str.strip()
    df["label"] = (df.relation == "entailment").astype(int)  # entailment = valid
    lookup = _corrected_lookup()
    fixed = []
    for premise, hyp_text, label in zip(df.premise.str.strip(), df.hypothesis.str.strip(), df.label):
        key = next((_key(premise, rel, *m.groups()) for pat, rel in PARAPHRASE_PATTERNS
                    if (m := pat.match(hyp_text))), None)
        fixed.append(lookup.get(key, label))
    return _with_columns(df[["input"]].assign(label=fixed))
