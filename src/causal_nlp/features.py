"""Stage 3 - Feature engineering: describe each (X, relation, Y) candidate by numbers.

Three feature groups, so the notebook can show what each reasoning step adds:
  * TEXT_FACT_FEATURES - read straight off the extracted statements.
  * PC_GRAPH_FEATURES  - read off the PDAG produced by the PC rules.
  * SUPPORT_FEATURES   - in how many of the equivalent DAGs the hypothesis holds.
All features are name-agnostic: renaming variables (A->Z, ...) cannot change them.
"""
from __future__ import annotations

from functools import lru_cache

import networkx as nx
import pandas as pd

from .causal_rules import PDAG, equivalent_dags, pc_from_premise, support
from .parsing import RELATIONS, Hypothesis, Premise, parse_hypothesis, parse_premise, split_input

RELATION_FEATURES = [f"rel_{r}" for r in RELATIONS]
TEXT_FACT_FEATURES = [
    "n_vars", "marginally_correlated", "adjacent", "n_sepsets", "min_sepset_size",
    "degree_x", "degree_y", "n_common_neighbours",
]
PC_GRAPH_FEATURES = [
    "edge_x_to_y", "edge_y_to_x", "edge_undirected",
    "x_ancestor_of_y", "y_ancestor_of_x",                    # via directed edges only
    "x_possible_ancestor_of_y", "y_possible_ancestor_of_x",  # undirected edges may point either way
    "n_common_children", "n_common_parents",
]
SUPPORT_FEATURES = ["support", "n_equivalent_dags"]
ALL_FEATURES = RELATION_FEATURES + TEXT_FACT_FEATURES + PC_GRAPH_FEATURES + SUPPORT_FEATURES


@lru_cache(maxsize=None)
def analyse_premise(premise_text: str) -> tuple[Premise, PDAG, nx.DiGraph, nx.DiGraph, list[nx.DiGraph]]:
    """Parse once per premise (many hypotheses share it) and build helper graphs."""
    premise = parse_premise(premise_text)
    pdag = pc_from_premise(premise)
    definite = nx.DiGraph(list(pdag.directed))
    definite.add_nodes_from(pdag.nodes)
    possible = definite.copy()
    for edge in pdag.undirected:
        a, b = tuple(edge)
        possible.add_edges_from([(a, b), (b, a)])
    return premise, pdag, definite, possible, equivalent_dags(pdag)


def pair_features(premise: Premise, pdag: PDAG, definite: nx.DiGraph,
                  possible: nx.DiGraph, dags: list[nx.DiGraph], hyp: Hypothesis) -> dict:
    x, y = hyp.x, hyp.y
    sepsets = premise.sepsets[frozenset((x, y))]
    feats = {f"rel_{r}": int(hyp.relation == r) for r in RELATIONS}
    feats.update({
        "n_vars": len(premise.variables),
        "marginally_correlated": int(frozenset((x, y)) in premise.correlated),
        "adjacent": int(pdag.adjacent(x, y)),
        "n_sepsets": len(sepsets),
        "min_sepset_size": min((len(s) for s in sepsets), default=-1),
        "degree_x": len(pdag.neighbours(x)),
        "degree_y": len(pdag.neighbours(y)),
        "n_common_neighbours": len(pdag.neighbours(x) & pdag.neighbours(y)),
        "edge_x_to_y": int((x, y) in pdag.directed),
        "edge_y_to_x": int((y, x) in pdag.directed),
        "edge_undirected": int(frozenset((x, y)) in pdag.undirected),
        "x_ancestor_of_y": int(nx.has_path(definite, x, y)),
        "y_ancestor_of_x": int(nx.has_path(definite, y, x)),
        "x_possible_ancestor_of_y": int(nx.has_path(possible, x, y)),
        "y_possible_ancestor_of_x": int(nx.has_path(possible, y, x)),
        "n_common_children": len(pdag.children(x) & pdag.children(y)),
        "n_common_parents": len(pdag.parents(x) & pdag.parents(y)),
        "support": support(dags, hyp.relation, x, y),
        "n_equivalent_dags": len(dags),
    })
    return feats


def build_feature_table(inputs: pd.Series, hypothesis_parser=parse_hypothesis) -> pd.DataFrame:
    """One row of features per Corr2Cause input. Unparseable hypotheses get parsed=0."""
    rows = []
    for text in inputs:
        premise_text, hyp_text = split_input(text)
        premise, pdag, definite, possible, dags = analyse_premise(premise_text)
        hyp = hypothesis_parser(hyp_text)
        if hyp is None:
            rows.append({"parsed": 0})
            continue
        rows.append({"parsed": 1, "relation": hyp.relation, "x": hyp.x, "y": hyp.y,
                     **pair_features(premise, pdag, definite, possible, dags, hyp)})
    return pd.DataFrame(rows, index=inputs.index)
