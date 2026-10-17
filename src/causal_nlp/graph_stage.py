"""Stage 5 - Graph stage: reviewed triples -> DAG -> Bayesian network -> do-queries.

Input is a "triples" table with one row per candidate relation:
    premise_id, relation, x, y, prob   (+ label for evaluation)
Both the Phase 1 (ML) and Phase 2 (DL) extractors produce this table, so this
module is shared and the comparison between them is fair.
"""
from __future__ import annotations

from itertools import product

import networkx as nx
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from .parsing import DIRECT, INDIRECT

BASE_RATE = 0.1        # P(variable = 1) when none of its causes is active
CAUSE_STRENGTH = 0.7   # chance that one active cause switches its effect on
EFFECT_THRESHOLD = 0.01


# ------------------------------------------------------------------ DAG building
def build_dag(nodes: list[str], scored_edges: list[tuple[str, str, float]]) -> tuple[nx.DiGraph, list]:
    """Add edges from most to least confident; skip any edge that would close a cycle.
    Returns the DAG and the rejected edges (so the notebook can show them)."""
    dag, rejected = nx.DiGraph(), []
    dag.add_nodes_from(nodes)
    for cause, effect, score in sorted(scored_edges, key=lambda e: -e[2]):
        if cause == effect or nx.has_path(dag, effect, cause):  # self loop or cycle
            rejected.append((cause, effect, score))
        else:
            dag.add_edge(cause, effect, weight=score)
    return dag, rejected


def dags_from_triples(triples: pd.DataFrame, nodes_by_premise: dict, threshold: float = 0.5) -> dict:
    """premise_id -> DAG built from the 'direct' triples predicted as valid."""
    direct = triples[(triples.relation == DIRECT) & (triples.prob >= threshold)]
    dags = {}
    for pid, nodes in nodes_by_premise.items():
        rows = direct[direct.premise_id == pid]
        dags[pid], _ = build_dag(nodes, list(zip(rows.x, rows.y, rows.prob)))
    return dags


def edge_metrics(dags: dict, triples: pd.DataFrame) -> dict:
    """Micro-averaged edge precision / recall / F1 against the gold direct-cause edges."""
    gold = triples[(triples.relation == DIRECT) & (triples.label == 1)]
    gold_edges = set(zip(gold.premise_id, gold.x, gold.y))
    pred_edges = {(pid, a, b) for pid, dag in dags.items() for a, b in dag.edges}
    tp = len(pred_edges & gold_edges)
    precision = tp / len(pred_edges) if pred_edges else 0.0
    recall = tp / len(gold_edges) if gold_edges else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"gold_edges": len(gold_edges), "predicted_edges": len(pred_edges), "correct": tp,
            "edge_precision": precision, "edge_recall": recall, "edge_f1": f1}


# ------------------------------------------------------------- Bayesian network
def noisy_or_network(dag: nx.DiGraph, base: float = BASE_RATE, strength: float = CAUSE_STRENGTH) -> DiscreteBayesianNetwork:
    """Binary BN where each active cause independently switches its effect on.
    P(child=1 | k active parents) = 1 - (1 - base) * (1 - strength) ** k"""
    bn = DiscreteBayesianNetwork(list(dag.edges))
    bn.add_nodes_from(dag.nodes)
    for node in dag.nodes:
        parents = sorted(dag.predecessors(node))
        p_on = [1 - (1 - base) * (1 - strength) ** sum(states)
                for states in product((0, 1), repeat=len(parents))]
        bn.add_cpds(TabularCPD(node, 2, [[1 - p for p in p_on], p_on],
                               evidence=parents or None, evidence_card=[2] * len(parents) or None))
    bn.check_model()
    return bn


def interventional_effect(bn: DiscreteBayesianNetwork, x: str, y: str) -> float:
    """Average causal effect  P(Y=1 | do(X=1)) - P(Y=1 | do(X=0)).
    Intervening on X can only change X's descendants, so other variables get 0."""
    if not nx.has_path(bn, x, y):
        return 0.0
    ci = CausalInference(bn)
    p = [ci.query([y], do={x: v}, show_progress=False).values[1] for v in (0, 1)]
    return float(p[1] - p[0])


def observational_effect(bn: DiscreteBayesianNetwork, x: str, y: str) -> float:
    """Plain association  P(Y=1 | X=1) - P(Y=1 | X=0)  (what correlation alone tells you)."""
    ve = VariableElimination(bn)
    p = [ve.query([y], evidence={x: v}, show_progress=False).values[1] for v in (0, 1)]
    return float(p[1] - p[0])


# ------------------------------------------------------- intervention queries
def known_intervention_queries(triples: pd.DataFrame, correlated_by_premise: dict) -> pd.DataFrame:
    """Queries 'does do(X) change Y?' whose answer is fixed by the gold labels / text:
      yes - X directly or indirectly causes Y (gold label 1)
      no  - Y causes X (an effect cannot cause its cause in a DAG), or
            X and Y are not even correlated (no association -> no causal effect)."""
    causes = triples[triples.relation.isin([DIRECT, INDIRECT]) & (triples.label == 1)]
    yes = set(zip(causes.premise_id, causes.x, causes.y))
    rows = [{"premise_id": p, "x": x, "y": y, "gold_effect": True} for p, x, y in sorted(yes)]
    for pid, (nodes, correlated) in correlated_by_premise.items():
        for x, y in product(nodes, nodes):
            if x == y or (pid, x, y) in yes:
                continue
            if (pid, y, x) in yes or frozenset((x, y)) not in correlated:
                rows.append({"premise_id": pid, "x": x, "y": y, "gold_effect": False})
    return pd.DataFrame(rows)


def answer_queries(queries: pd.DataFrame, dags: dict) -> pd.DataFrame:
    """Build a noisy-OR BN on each predicted DAG and answer every query with do-calculus."""
    networks = {pid: noisy_or_network(dag) for pid, dag in dags.items()}
    effects = [interventional_effect(networks[p], x, y) for p, x, y in zip(queries.premise_id, queries.x, queries.y)]
    return queries.assign(effect=effects, predicted_effect=[e > EFFECT_THRESHOLD for e in effects])
