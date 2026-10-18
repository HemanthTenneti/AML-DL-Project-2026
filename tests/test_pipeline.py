import networkx as nx
import pytest

from causal_nlp.causal_rules import equivalent_dags, pc_from_premise, relation_holds, support
from causal_nlp.graph_stage import build_dag, interventional_effect, noisy_or_network, observational_effect
from causal_nlp.parsing import COLLIDER, CONFOUNDER, DIRECT, INDIRECT, parse_hypothesis, parse_premise

COLLIDER_TEXT = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations "
                 "among these 3 variables are as follows: A correlates with C. B correlates with C. "
                 "However, A is independent of B.")
CHAIN_TEXT = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations "
              "among these 3 variables are as follows: A correlates with B. A correlates with C. "
              "B correlates with C. However, A and C are independent given B.")


def test_parse_premise_extracts_facts():
    p = parse_premise(COLLIDER_TEXT)
    assert p.variables == ["A", "B", "C"]
    assert frozenset("AC") in p.correlated and frozenset("AB") not in p.correlated
    assert p.sepsets[frozenset("AB")] == [frozenset()]


def test_parse_conditional_independence():
    p = parse_premise(CHAIN_TEXT)
    assert p.sepsets[frozenset("AC")] == [frozenset("B")]


@pytest.mark.parametrize("text, expected", [
    ("A directly causes B.", (DIRECT, "A", "B")),
    ("C causes something else which causes A.", (INDIRECT, "C", "A")),
    ("B is a cause for D, but not a direct one.", (INDIRECT, "B", "D")),
    ("There exists at least one collider (i.e., common effect) of A and C.", (COLLIDER, "A", "C")),
    ("There exists at least one confounder (i.e., common cause) of B and E.", (CONFOUNDER, "B", "E")),
])
def test_parse_hypothesis(text, expected):
    h = parse_hypothesis(text)
    assert (h.relation, h.x, h.y) == expected


def test_unseen_phrasing_is_not_parsed():
    assert parse_hypothesis("A directly affects B.") is None


def test_pc_finds_v_structure():
    pdag = pc_from_premise(parse_premise(COLLIDER_TEXT))
    assert pdag.directed == {("A", "C"), ("B", "C")}
    assert not pdag.undirected


def test_chain_is_unoriented_with_three_equivalent_dags():
    pdag = pc_from_premise(parse_premise(CHAIN_TEXT))
    assert pdag.undirected == {frozenset("AB"), frozenset("BC")}
    # A->B->C, A<-B<-C, A<-B->C are equivalent; A->B<-C is not (it is a v-structure).
    assert len(equivalent_dags(pdag)) == 3


def test_support_values():
    dags = equivalent_dags(pc_from_premise(parse_premise(COLLIDER_TEXT)))
    assert support(dags, DIRECT, "A", "C") == 1.0
    assert support(dags, COLLIDER, "A", "B") == 1.0
    assert support(dags, CONFOUNDER, "A", "B") == 0.0


def test_relation_semantics():
    dag = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "M"), ("M", "Y")])
    assert relation_holds(dag, CONFOUNDER, "X", "Y")
    assert relation_holds(dag, INDIRECT, "X", "Y")
    assert not relation_holds(dag, DIRECT, "X", "Y")


def test_build_dag_rejects_cycle_and_self_loop():
    dag, rejected = build_dag(["A", "B", "C"], [("A", "B", 0.9), ("B", "C", 0.8), ("C", "A", 0.6), ("A", "A", 0.5)])
    assert nx.is_directed_acyclic_graph(dag)
    assert {(c, e) for c, e, _ in rejected} == {("C", "A"), ("A", "A")}


def test_confounding_makes_association_bigger_than_causation():
    bn = noisy_or_network(nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "Y")]))
    assert interventional_effect(bn, "X", "Y") > 0
    assert observational_effect(bn, "X", "Y") > interventional_effect(bn, "X", "Y")
    assert interventional_effect(bn, "Y", "X") == 0.0
