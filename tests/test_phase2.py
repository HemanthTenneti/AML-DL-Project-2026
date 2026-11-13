"""Phase 2 unit tests: no model training, so they run in seconds."""
import random

import pytest

from causal_nlp.dl_common import length_grouped_batches
from causal_nlp.dl_end2end import premise_variables, rename_variables
from causal_nlp.dl_parser import (LABEL_ID, LABELS, ORIGINAL_TEMPLATES, check_no_test_leak,
                                  decode, make_examples, template_label, training_templates, variables_in_order)
from causal_nlp.parsing import COLLIDER, DIRECT, INDIRECT, Hypothesis, parse_hypothesis, parse_premise

PREMISE = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations "
           "among these 3 variables are as follows: A correlates with C. B correlates with C. "
           "However, A is independent of B.")


def test_premise_variables_in_header_order():
    assert premise_variables(PREMISE) == ["A", "B", "C"]


def test_renaming_is_consistent_and_keeps_the_facts():
    premise, hyp = rename_variables(PREMISE, "A directly causes C.", random.Random(0))
    old, new = parse_premise(PREMISE), parse_premise(premise)
    mapping = dict(zip(old.variables, new.variables))
    assert len(set(new.variables)) == 3
    assert {frozenset(mapping[v] for v in pair) for pair in old.correlated} == new.correlated
    h = parse_hypothesis(hyp)
    assert (h.x, h.y) == (mapping["A"], mapping["C"])


@pytest.mark.parametrize("template, relation, expected", [
    ("{x} directly causes {y}.", DIRECT, "direct"),
    ("{y} is directly caused by {x}.", DIRECT, "direct_reversed"),
    ("{y} is an indirect effect of {x}.", INDIRECT, "indirect_reversed"),
    ("{x} and {y} share a common effect.", COLLIDER, "collider"),
])
def test_template_label_encodes_direction(template, relation, expected):
    assert template_label(template, relation) == expected


@pytest.mark.parametrize("label, text, expected", [
    ("direct", "A directly causes B.", Hypothesis(DIRECT, "A", "B")),
    ("direct_reversed", "B is directly caused by A.", Hypothesis(DIRECT, "A", "B")),
    ("indirect_reversed", "Q is an indirect effect of P.", Hypothesis(INDIRECT, "P", "Q")),
    ("collider", "There is a common effect of D and E.", Hypothesis(COLLIDER, "D", "E")),
    ("direct", "A directly causes A.", None),          # self-loop: only one variable
    ("direct", "Nothing here.", None),
])
def test_decode(label, text, expected):
    assert decode(label, text) == expected


def test_every_template_has_exactly_two_variables():
    train, val = training_templates()
    for templates in (train, val, ORIGINAL_TEMPLATES):
        for items in templates.values():
            for template in items:
                assert sorted(variables_in_order(template.format(x="P", y="Q"))) == ["P", "Q"], template


def test_grammar_never_uses_the_paraphrase_test_wording():
    train, val = training_templates()
    check_no_test_leak(train)
    check_no_test_leak(val)


def test_leak_check_catches_a_test_sentence():
    with pytest.raises(ValueError):
        check_no_test_leak({DIRECT: ["{x} directly affects {y}."]})


def test_validation_verbs_are_unseen_in_training():
    train, val = training_templates()
    train_text = " ".join(t for items in train.values() for t in items)
    assert "trigger" not in train_text and "generat" not in train_text
    assert all(val.values())


def test_make_examples_labels_match_the_regex_parser_on_original_templates():
    df = make_examples(ORIGINAL_TEMPLATES, n_per_relation=20, seed=0)
    for text, label, relation, x, y in zip(df.text, df.label, df.relation, df.x, df.y):
        assert parse_hypothesis(text) == Hypothesis(relation, x, y)
        assert decode(LABELS[label], text) == Hypothesis(relation, x, y)
    assert set(df.label) == {LABEL_ID[r] for r in ("direct", "indirect", "collider", "confounder")}


def test_length_grouped_batches_cover_every_example_once():
    lengths = [random.Random(i).randint(5, 500) for i in range(1000)]
    batches = length_grouped_batches(lengths, 16, random.Random(0))
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(1000))
    assert max(len(b) for b in batches) == 16
