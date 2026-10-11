import pytest

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
