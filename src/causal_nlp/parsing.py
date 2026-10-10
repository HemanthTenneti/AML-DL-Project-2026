"""Stage 1 - Language layer: turn Corr2Cause text into structured facts.

Phase 1 uses simple regular-expression rules (the "AML baseline" extractor).
Phase 2 can swap `parse_hypothesis` for a DistilBERT-based parser without
touching anything downstream, because both return the same `Hypothesis` tuple.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

# The four causal relation types we reason about (our small "entity schema").
DIRECT, INDIRECT, COLLIDER, CONFOUNDER = "direct", "indirect", "collider", "confounder"
RELATIONS = (DIRECT, INDIRECT, COLLIDER, CONFOUNDER)

VAR = r"([A-Z])"  # variables are single capital letters (A..F, or V..Z after refactoring)


@dataclass(frozen=True)
class Hypothesis:
    relation: str  # one of RELATIONS
    x: str         # cause slot (or first variable for symmetric relations)
    y: str         # effect slot (or second variable)


@dataclass
class Premise:
    variables: list[str]
    correlated: set[frozenset] = field(default_factory=set)      # marginally dependent pairs
    sepsets: dict[frozenset, list[frozenset]] = field(default_factory=dict)  # pair -> sets S with X _||_ Y | S

    def pairs(self):
        return [frozenset(p) for p in combinations(self.variables, 2)]


# --------------------------------------------------------------------------- premise
_HEADER = re.compile(r"closed system of (\d+) variables?, (.+?)\. All the statistical")
_CORR = re.compile(rf"{VAR} correlates with {VAR}\.")
_MARG_INDEP = re.compile(rf"{VAR} is independent of {VAR}\.")
_COND_INDEP = re.compile(rf"{VAR} and {VAR} are independent given ([A-Z ,and]+?)\.")


def _split_var_list(text: str) -> list[str]:
    """'A, B, C and D' -> ['A', 'B', 'C', 'D']"""
    return re.findall(r"\b[A-Z]\b", text)


def parse_premise(text: str) -> Premise:
    header = _HEADER.search(text)
    if header is None:
        raise ValueError(f"Unrecognised premise: {text[:80]}...")
    n_vars, var_list = int(header.group(1)), _split_var_list(header.group(2))
    if len(var_list) != n_vars:
        raise ValueError(f"Expected {n_vars} variables, found {var_list}")

    premise = Premise(variables=var_list)
    premise.sepsets = {pair: [] for pair in premise.pairs()}
    body = text[header.end():]

    for a, b in _CORR.findall(body):
        premise.correlated.add(frozenset((a, b)))
    for a, b in _MARG_INDEP.findall(body):
        premise.sepsets[frozenset((a, b))].append(frozenset())
    for a, b, given in _COND_INDEP.findall(body):
        premise.sepsets[frozenset((a, b))].append(frozenset(_split_var_list(given)))
    return premise


# ------------------------------------------------------------------------ hypothesis
# Each pattern maps a sentence template to (relation, which group is x, which is y).
HYPOTHESIS_PATTERNS = [
    (re.compile(rf"^{VAR} directly causes {VAR}\.$"), DIRECT, 1, 2),
    (re.compile(rf"^{VAR} causes something else which causes {VAR}\.$"), INDIRECT, 1, 2),
    (re.compile(rf"^{VAR} is a cause for {VAR}, but not a direct one\.$"), INDIRECT, 1, 2),
    (re.compile(rf"^There exists at least one collider \(i\.e\., common effect\) of {VAR} and {VAR}\.$"), COLLIDER, 1, 2),
    (re.compile(rf"^There exists at least one confounder \(i\.e\., common cause\) of {VAR} and {VAR}\.$"), CONFOUNDER, 1, 2),
]


def parse_hypothesis(text: str) -> Hypothesis | None:
    """Return the structured hypothesis, or None if no rule matches (unseen phrasing)."""
    text = text.strip()
    for pattern, relation, gx, gy in HYPOTHESIS_PATTERNS:
        m = pattern.match(text)
        if m:
            return Hypothesis(relation, m.group(gx), m.group(gy))
    return None


def split_input(text: str) -> tuple[str, str]:
    """Corr2Cause 'input' column -> (premise, hypothesis)."""
    premise, hypothesis = text.split("\nHypothesis:", 1)
    return premise.replace("Premise:", "", 1).strip(), hypothesis.strip()
