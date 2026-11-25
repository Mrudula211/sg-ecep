"""Tests for the grammar, the baseline parser and the error-correcting parser."""

from __future__ import annotations

import pytest

from sgecep.earley import EarleyParser
from sgecep.ec_earley import ErrorCorrectingEarleyParser
from sgecep.edits import CostModel, EditKind
from sgecep.grammar import Grammar, Lexicon, check_coverage

from tests.conftest import tokens


# -- grammar ----------------------------------------------------------------


def test_grammar_and_lexicon_agree(grammar: Grammar, lexicon: Lexicon) -> None:
    """Every pre-terminal is fillable and every lexicon tag is reachable."""
    assert check_coverage(grammar, lexicon) == []


def test_preterminals_are_disjoint_from_nonterminals(grammar: Grammar) -> None:
    assert not (grammar.preterminals & grammar.nonterminals)


def test_grammar_parsing_of_alternatives() -> None:
    grammar = Grammar.from_string("S -> A B | A\nA -> x\nB -> y")
    assert len(grammar.rules) == 4
    assert {tuple(r.rhs) for r in grammar.rules_for("S")} == {("A", "B"), ("A",)}


def test_epsilon_production_is_nullable() -> None:
    grammar = Grammar.from_string("S -> A B\nA -> EPS\nB -> y")
    assert "A" in grammar.nullable()
    assert "B" not in grammar.nullable()


def test_lexicon_reverse_index(lexicon: Lexicon) -> None:
    assert "DTS" in lexicon.tags("the")
    assert "the" in lexicon.words("DTS")
    assert lexicon.lemma("students") == "student"
    assert not lexicon.knows("zzzz")


# -- baseline Earley ---------------------------------------------------------


GRAMMATICAL = [
    "the student reads the book",
    "the students read the book",
    "they eat the pizza",
    "john gave mary a book",
    "dogs run",
    "open the door",
    "the teacher is happy",
    "the hungry dog quickly ate the fresh sandwich",
    "he thinks that the student reads the book",
    "the student ate the pizza in the kitchen",
]

UNGRAMMATICAL = [
    "the students reads the book",     # agreement
    "student reads the book",          # missing determiner
    "the the dog ate the pizza",       # duplicated determiner
    "the man slept big",               # subcategorisation: sleep is intransitive
    "the dog ate yesterday the car",   # adjunct in a complement slot
    "the student reads the quickly",   # adverb where a noun is required
]


@pytest.mark.parametrize("sentence", GRAMMATICAL)
def test_baseline_accepts_grammatical(parser: EarleyParser, sentence: str) -> None:
    assert parser.recognise(tokens(sentence))
    assert parser.parse(tokens(sentence), limit=1)


@pytest.mark.parametrize("sentence", UNGRAMMATICAL)
def test_baseline_rejects_ungrammatical(parser: EarleyParser, sentence: str) -> None:
    assert not parser.recognise(tokens(sentence))
    assert parser.parse(tokens(sentence)) == []


def test_baseline_handles_left_recursion(parser: EarleyParser) -> None:
    """`VPS -> VPS ADVP` is left-recursive; Earley must not loop on it."""
    assert parser.recognise(tokens("the dog ate the pizza quickly slowly"))


def test_pp_attachment_is_ambiguous(parser: EarleyParser) -> None:
    trees = parser.parse(tokens("she saw the man with the telescope"), limit=10)
    assert len(trees) >= 2, "the classic PP-attachment ambiguity should survive"


def test_parse_tree_yields_the_input(parser: EarleyParser) -> None:
    sentence = "the student reads the book"
    tree = parser.parse(tokens(sentence), limit=1)[0]
    assert tree.yield_words() == tokens(sentence)


# -- cost model --------------------------------------------------------------


def test_cost_model_rejects_free_insertion() -> None:
    with pytest.raises(ValueError):
        CostModel(insert_function=0.0).validate()


def test_cost_model_orders_function_below_content() -> None:
    costs = CostModel()
    assert costs.insert_cost("DTS") < costs.insert_cost("NNS")
    assert costs.delete_cost({"DTS"}) < costs.delete_cost({"NNS"})


def test_unknown_token_is_cheap_to_delete() -> None:
    costs = CostModel()
    assert costs.delete_cost(set()) == costs.delete_unknown


# -- error-correcting Earley -------------------------------------------------


def test_grammatical_input_costs_nothing(
    ec_parser: ErrorCorrectingEarleyParser,
) -> None:
    assert ec_parser.best_cost(tokens("the student reads the book")) == 0.0


def test_agreement_error_is_repaired_by_one_inflection(
    ec_parser: ErrorCorrectingEarleyParser,
) -> None:
    cost = ec_parser.best_cost(tokens("the students reads the book"))
    assert cost == pytest.approx(CostModel().inflect)


def test_missing_determiner_is_repaired_by_one_insertion(
    ec_parser: ErrorCorrectingEarleyParser,
) -> None:
    cost = ec_parser.best_cost(tokens("student reads the book"))
    assert cost == pytest.approx(CostModel().insert_function)


def test_spurious_token_is_repaired_by_one_deletion(
    ec_parser: ErrorCorrectingEarleyParser,
) -> None:
    cost = ec_parser.best_cost(tokens("the the dog ate the pizza"))
    assert cost == pytest.approx(CostModel().delete_function)


def test_enumeration_reaches_the_chart_minimum(
    ec_parser: ErrorCorrectingEarleyParser,
) -> None:
    """The k-best walk must actually recover the cheapest chart derivation.

    Regression test: ordering edges by their own cost rather than by a lower
    bound let the per-item beam discard the cheapest repair, so `best_cost`
    and `repairs` disagreed.
    """
    for sentence in UNGRAMMATICAL:
        toks = tokens(sentence)
        best = ec_parser.best_cost(toks)
        repairs = ec_parser.repairs(toks, k=5)
        assert repairs, f"no repair found for {sentence!r}"
        assert best is not None
        assert repairs[0].syntactic_cost == pytest.approx(best), sentence


def test_repairs_are_returned_in_cost_order(
    ec_parser: ErrorCorrectingEarleyParser,
) -> None:
    repairs = ec_parser.repairs(tokens("the the students reads the book"), k=6)
    costs = [r.syntactic_cost for r in repairs]
    assert costs == sorted(costs)


def test_agreement_error_yields_a_genuine_tie(
    ec_parser: ErrorCorrectingEarleyParser,
) -> None:
    """Either the subject or the verb can be fixed, at identical cost.

    This tie is the whole reason the reranker exists, so it is asserted
    rather than merely relied upon.
    """
    repairs = ec_parser.repairs(tokens("the students reads the book"), k=6)
    cheapest = min(r.syntactic_cost for r in repairs)
    tied = [r for r in repairs if r.syntactic_cost == pytest.approx(cheapest)]
    assert len(tied) >= 2


def test_slots_survive_tree_traversal(
    ec_parser: ErrorCorrectingEarleyParser,
) -> None:
    """Unfilled slots must appear as leaves.

    Regression test: `is_leaf` once tested `word is not None`, so slots were
    treated as childless internal nodes and vanished from the yield.
    """
    repair = ec_parser.repairs(tokens("the students reads the book"), k=1)[0]
    assert repair.slot_count() >= 1
    assert any(leaf.is_slot for leaf in repair.tree.leaves())
    assert len(repair.surface()) == len(repair.tree.leaves())


def test_edits_record_their_position_and_kind(
    ec_parser: ErrorCorrectingEarleyParser,
) -> None:
    repair = ec_parser.repairs(tokens("the the dog ate the pizza"), k=1)[0]
    deletions = [e for e in repair.edits if e.kind is EditKind.DELETE]
    assert deletions
    assert deletions[0].original == "the"


def test_cost_budget_is_respected(
    grammar: Grammar, lexicon: Lexicon
) -> None:
    tight = ErrorCorrectingEarleyParser(grammar, lexicon, max_cost=0.2)
    assert tight.repairs(tokens("student reads the book")) == []


def test_insertion_terminates_on_hopeless_input(
    ec_parser: ErrorCorrectingEarleyParser,
) -> None:
    """Insertion consumes no input, so only the positive cost bounds it."""
    assert ec_parser.repairs(tokens("quickly quickly quickly"), k=3) is not None
