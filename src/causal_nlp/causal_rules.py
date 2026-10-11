"""Stage 2 - Symbolic layer: the PC algorithm's rules applied to extracted facts.

Given a complete list of (in)dependence statements, the PC algorithm recovers the
causal graph up to its Markov equivalence class as a PDAG (some edges directed,
some left undirected because the data cannot tell their direction):

  1. Skeleton     - X and Y are joined iff no conditioning set makes them independent.
  2. V-structures - for X - Z - Y with X, Y not joined: if Z is in no separating set
                    of (X, Y), then Z is a common effect:  X -> Z <- Y.
  3. Meek rules   - propagate arrows that are forced to avoid new v-structures/cycles.

A PDAG stands for several DAGs ("Markov equivalent" - they imply exactly the same
(in)dependences). A hypothesis is only *valid* if it holds in every one of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import networkx as nx

from .parsing import COLLIDER, CONFOUNDER, DIRECT, INDIRECT, Premise


@dataclass
class PDAG:
    nodes: list[str]
    directed: set[tuple[str, str]] = field(default_factory=set)  # (a, b) means a -> b
    undirected: set[frozenset] = field(default_factory=set)      # {a, b} means a - b

    def adjacent(self, a: str, b: str) -> bool:
        return (a, b) in self.directed or (b, a) in self.directed or frozenset((a, b)) in self.undirected

    def neighbours(self, a: str) -> set[str]:
        return {b for b in self.nodes if b != a and self.adjacent(a, b)}

    def parents(self, a: str) -> set[str]:
        return {p for p, c in self.directed if c == a}

    def children(self, a: str) -> set[str]:
        return {c for p, c in self.directed if p == a}

    def orient(self, a: str, b: str) -> None:
        """Turn the undirected edge a - b into a -> b."""
        self.undirected.discard(frozenset((a, b)))
        self.directed.add((a, b))


def skeleton(premise: Premise) -> PDAG:
    pdag = PDAG(nodes=list(premise.variables))
    for pair in premise.pairs():
        if not premise.sepsets[pair]:  # never independent -> direct link
            pdag.undirected.add(pair)
    return pdag


def orient_v_structures(pdag: PDAG, premise: Premise) -> PDAG:
    for x, y in combinations(pdag.nodes, 2):
        if pdag.adjacent(x, y):
            continue
        sepsets = premise.sepsets[frozenset((x, y))]
        for z in pdag.neighbours(x) & pdag.neighbours(y):
            if all(z not in s for s in sepsets):
                pdag.orient(x, z)
                pdag.orient(y, z)
    return pdag


def apply_meek_rules(pdag: PDAG) -> PDAG:
    changed = True
    while changed:
        changed = False
        for edge in list(pdag.undirected):
            a, b = tuple(edge)
            for u, v in ((a, b), (b, a)):
                if frozenset((u, v)) not in pdag.undirected:
                    break
                if _forced(pdag, u, v):
                    pdag.orient(u, v)
                    changed = True
    return pdag


def _forced(pdag: PDAG, u: str, v: str) -> bool:
    """Is the undirected edge u - v forced to be u -> v by Meek rules R1-R3?"""
    # R1: w -> u - v with w, v not adjacent   =>  u -> v   (else new v-structure at u)
    if any(not pdag.adjacent(w, v) for w in pdag.parents(u) if w != v):
        return True
    # R2: u -> w -> v and u - v               =>  u -> v   (else a cycle)
    if pdag.children(u) & pdag.parents(v):
        return True
    # R3: u - w1 -> v, u - w2 -> v, w1, w2 not adjacent  =>  u -> v
    undirected_nbrs = {w for w in pdag.nodes if frozenset((u, w)) in pdag.undirected}
    candidates = sorted(undirected_nbrs & pdag.parents(v))
    return any(not pdag.adjacent(w1, w2) for w1, w2 in combinations(candidates, 2))


def pc_from_premise(premise: Premise) -> PDAG:
    """Run all three PC steps on the extracted (in)dependence facts."""
    return apply_meek_rules(orient_v_structures(skeleton(premise), premise))


# ------------------------------------------------------------ equivalent DAGs
def _v_structures(dag: nx.DiGraph, adjacency: nx.Graph) -> set[tuple[str, str, str]]:
    found = set()
    for z in dag.nodes:
        for a, b in combinations(sorted(dag.predecessors(z)), 2):
            if not adjacency.has_edge(a, b):
                found.add((a, z, b))
    return found


def equivalent_dags(pdag: PDAG) -> list[nx.DiGraph]:
    """All DAGs the PDAG stands for: try both directions for every undirected edge,
    keep those with no cycle and no extra v-structure (they fit the facts equally well)."""
    undirected = [tuple(sorted(e)) for e in pdag.undirected]
    base = nx.DiGraph(list(pdag.directed))
    base.add_nodes_from(pdag.nodes)
    adjacency = base.to_undirected()
    adjacency.add_edges_from(undirected)
    required = _v_structures(base, adjacency)

    dags = []
    for flips in product((False, True), repeat=len(undirected)):
        dag = base.copy()
        dag.add_edges_from((b, a) if flip else (a, b) for (a, b), flip in zip(undirected, flips))
        if nx.is_directed_acyclic_graph(dag) and _v_structures(dag, adjacency) == required:
            dags.append(dag)
    return dags


# --------------------------------------------------- relation semantics in one DAG
def _reachable_avoiding(dag: nx.DiGraph, start: str, avoid: str, reverse: bool = False) -> set[str]:
    """Descendants (or ancestors) of `start` along paths that do not pass through `avoid`."""
    view = nx.subgraph_view(dag, filter_node=lambda n: n != avoid)
    return nx.ancestors(view, start) if reverse else nx.descendants(view, start)


def relation_holds(dag: nx.DiGraph, relation: str, x: str, y: str) -> bool:
    if relation == DIRECT:      # X -> Y
        return dag.has_edge(x, y)
    if relation == INDIRECT:    # X -> Z -> ... -> Y  (through at least one mediator)
        return any(nx.has_path(dag, z, y) for z in dag.successors(x) if z != y)
    if relation == COLLIDER:    # X -> ... -> Z <- ... <- Y  (a common effect)
        return bool(_reachable_avoiding(dag, x, y) & _reachable_avoiding(dag, y, x))
    if relation == CONFOUNDER:  # X <- ... <- Z -> ... -> Y  (a common cause)
        return bool(_reachable_avoiding(dag, x, y, reverse=True) & _reachable_avoiding(dag, y, x, reverse=True))
    raise ValueError(relation)


def support(dags: list[nx.DiGraph], relation: str, x: str, y: str) -> float:
    """Fraction of equivalent DAGs in which the relation holds (1.0 = certain)."""
    return sum(relation_holds(d, relation, x, y) for d in dags) / len(dags)
