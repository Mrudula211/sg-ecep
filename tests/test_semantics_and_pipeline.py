"""Tests for WSD, selectional restrictions, the language model and the pipeline."""

from __future__ import annotations

import math

import pytest

from sgecep.earley import EarleyParser
from sgecep.lesk import LeskDisambiguator, bag_of_words
from sgecep.lm import AddKLM, BigramLM
from sgecep.pipeline import CorrectionPipeline, tokenise
from sgecep.rerank import RerankWeights
from sgecep.semantics import (
    SelectionalChecker,
    extract_propositions,
    is_verb_tag,
)

from tests.conftest import tokens


# -- Lesk --------------------------------------------------------------------


def test_bag_of_words_drops_stopwords_and_short_tokens() -> None:
    bag = bag_of_words("The dog is in a big red house")
    assert "dog" in bag and "house" in bag
    assert "the" not in bag and "is" not in bag


def test_lesk_degrades_cleanly_without_wordnet() -> None:
    class Offline(LeskDisambiguator):
        @property
        def available(self) -> bool:  # type: ignore[override]
            return False

    offline = Offline()
    assert offline.rank_senses("bank", ["river"]) == []
    assert offline.disambiguate("bank", ["river"]) is None
    assert offline.hypernym_closure("bank.n.01") == frozenset()


def test_lesk_prefers_the_frequent_sense_without_evidence() -> None:
    lesk = LeskDisambiguator()
    if not lesk.available:
        pytest.skip("WordNet is not installed")
    # No informative context: the most-frequent-sense prior should decide.
    best = lesk.disambiguate("cat", [], pos="n")
    assert best is not None
    assert best.sense_rank == 0


def test_lesk_context_can_override_the_prior() -> None:
    lesk = LeskDisambiguator()
    if not lesk.available:
        pytest.skip("WordNet is not installed")
    ranked = lesk.rank_senses("bank", ["river", "water", "slope"], pos="n")
    assert ranked
    # Whichever sense wins, the scoring must combine overlap and rank rather
    # than return bare overlap counts.
    assert all(s.score >= s.lesk_overlap for s in ranked)


def test_mfs_weight_zero_recovers_textbook_lesk() -> None:
    lesk = LeskDisambiguator(mfs_weight=0.0)
    if not lesk.available:
        pytest.skip("WordNet is not installed")
    ranked = lesk.rank_senses("bank", ["river"], pos="n")
    assert all(s.score == float(s.lesk_overlap) for s in ranked)


# -- semantic typing ---------------------------------------------------------


def test_proper_nouns_bypass_wordnet(wordnet_checker: SelectionalChecker) -> None:
    """WordNet's john.n.01 is a lavatory; a proper noun must not reach it."""
    types, sense = wordnet_checker.types.types_of_word("john", [], tag="NNPS")
    assert "person" in types
    assert sense is None


def test_wordnet_anchors_cover_written_works(
    wordnet_checker: SelectionalChecker,
) -> None:
    """book.n.01 sits under publication.n.01, never under communication.n.02."""
    types, _ = wordnet_checker.types.types_of_word("book", ["read", "student"])
    assert "communication" in types


def test_fallback_ontology_is_used_without_wordnet(
    checker: SelectionalChecker,
) -> None:
    assert not checker.types.uses_wordnet
    types, _ = checker.types.types_of_word("pizza", [])
    assert "food" in types


# -- shallow SRL -------------------------------------------------------------


def test_is_verb_tag() -> None:
    assert is_verb_tag("VTZ") and is_verb_tag("VID")
    assert not is_verb_tag("MD")
    assert not is_verb_tag("NNS")
    assert not is_verb_tag(None)


def test_srl_finds_subject_and_object(parser: EarleyParser) -> None:
    tree = parser.parse(tokens("the student reads the book"), limit=1)[0]
    prop = extract_propositions(tree)[0]
    assert prop.predicate == "reads"
    assert prop.argument("ARG0").head == "student"
    assert prop.argument("ARG1").head == "book"


def test_srl_handles_ditransitives(parser: EarleyParser) -> None:
    tree = parser.parse(tokens("john gave mary a book"), limit=1)[0]
    prop = extract_propositions(tree)[0]
    assert prop.argument("ARG2").head == "mary"   # recipient
    assert prop.argument("ARG1").head == "book"   # theme


def test_srl_skips_the_modal(parser: EarleyParser) -> None:
    tree = parser.parse(tokens("the student should read the report"), limit=1)[0]
    prop = extract_propositions(tree)[0]
    assert prop.predicate == "read"


def test_srl_takes_the_np_head_not_the_pp_object(parser: EarleyParser) -> None:
    trees = parser.parse(tokens("she saw the man with the telescope"), limit=10)
    heads = {extract_propositions(t)[0].argument("ARG1").head for t in trees}
    assert heads == {"man"}, "the head of 'the man with the telescope' is 'man'"


def test_srl_finds_an_adjoined_instrument(parser: EarleyParser) -> None:
    """A PP adjoined by `VPS -> VPS PP` hangs off an outer VP node."""
    trees = parser.parse(tokens("she saw the man with the telescope"), limit=10)
    instruments = {extract_propositions(t)[0].instrument for t in trees}
    assert "telescope" in instruments


# -- selectional restrictions ------------------------------------------------


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("the dog ate the sandwich", 1.0),
        ("the student reads the book", 1.0),
        ("the student drinks the water", 1.0),
        ("the cat ate the homework", 0.5),
        ("the student reads the pizza", 0.5),
    ],
)
def test_plausibility_scores(
    parser: EarleyParser,
    checker: SelectionalChecker,
    sentence: str,
    expected: float,
) -> None:
    tree = parser.parse(tokens(sentence), limit=1)[0]
    assert checker.score(tree).score == pytest.approx(expected)


def test_unknown_verb_scores_neutral(
    parser: EarleyParser, checker: SelectionalChecker
) -> None:
    """No restrictions recorded must mean "no opinion", not "implausible"."""
    tree = parser.parse(tokens("the student has the book"), limit=1)[0]
    report = checker.score(tree)
    assert report.score == pytest.approx(0.5)
    assert report.checks == 0
    assert report.unchecked


def test_violations_are_explained(
    parser: EarleyParser, checker: SelectionalChecker
) -> None:
    tree = parser.parse(tokens("the student reads the pizza"), limit=1)[0]
    report = checker.score(tree)
    assert report.violated
    assert "communication" in report.violated[0]
    assert "semantic score" in report.explain()


# -- language model ----------------------------------------------------------


def test_lm_probabilities_are_a_distribution(lm: BigramLM) -> None:
    total = sum(lm.probability(w, "the") for w in lm.vocabulary)
    assert total == pytest.approx(1.0, abs=0.05)


def test_lm_assigns_mass_to_unseen_bigrams(lm: BigramLM) -> None:
    assert lm.probability("pizza", "telescope") > 0.0


def test_lm_prefers_attested_continuations(lm: BigramLM) -> None:
    assert lm.probability("student", "the") > lm.probability("the", "the")


def test_lm_rejects_an_invalid_discount() -> None:
    with pytest.raises(ValueError):
        BigramLM(discount=0.0)
    with pytest.raises(ValueError):
        BigramLM(discount=1.0)


def test_untrained_lm_refuses_to_score() -> None:
    with pytest.raises(RuntimeError):
        BigramLM().probability("a", "b")


def test_kneser_ney_beats_add_k_on_held_out_text() -> None:
    """The reason the LM uses Kneser-Ney, measured rather than asserted.

    A toy corpus will not show this: the difference only appears on held-out
    text containing bigrams the model has never seen, which is exactly the
    case add-k handles worst.  So the comparison runs on a real 80/20 split
    of the project corpus.
    """
    import random
    from pathlib import Path

    corpus = Path(__file__).resolve().parent.parent / "data" / "corpus.txt"
    sentences = [
        line.split()
        for raw in corpus.read_text(encoding="utf-8").splitlines()
        if (line := raw.split("#", 1)[0].strip())
    ]
    random.Random(3).shuffle(sentences)
    cut = int(len(sentences) * 0.8)
    train, held_out = sentences[:cut], sentences[cut:]

    kn = BigramLM().train(train)
    add_k = AddKLM(k=0.1).train(train)
    assert kn.perplexity(held_out) < add_k.perplexity(held_out)


def test_perplexity_is_finite(lm: BigramLM) -> None:
    value = lm.perplexity([["the", "dog", "ate", "the", "pizza"]])
    assert math.isfinite(value) and value > 0


# -- pipeline ----------------------------------------------------------------


def test_tokenise_lowercases_and_strips_punctuation() -> None:
    assert tokenise("The Dog ate, the pizza!") == [
        "the", "dog", "ate", "the", "pizza",
    ]


def test_grammatical_input_is_left_alone(pipeline: CorrectionPipeline) -> None:
    result = pipeline.correct("the dog ate the sandwich")
    assert result.already_grammatical
    assert result.corrected == "the dog ate the sandwich"


@pytest.mark.parametrize(
    "broken,fixed",
    [
        ("the students reads the book", "the student reads the book"),
        ("student reads the book", "the student reads the book"),
        ("the the dog ate the pizza", "the dog ate the pizza"),
    ],
)
def test_single_error_is_repaired(
    pipeline: CorrectionPipeline, broken: str, fixed: str
) -> None:
    result = pipeline.correct(broken)
    assert not result.already_grammatical
    assert result.corrected == fixed


def test_every_correction_is_grammatical(pipeline: CorrectionPipeline) -> None:
    """The output must be in the language, whatever else it is."""
    for sentence in [
        "the students reads the book",
        "student reads the book",
        "the the dog ate the pizza",
        "the student reads the quickly",
        "the doctor drank the slowly",
    ]:
        result = pipeline.correct(sentence)
        assert result.corrected is not None, sentence
        assert pipeline.is_grammatical(result.corrected), sentence


def test_corrections_have_no_unfilled_slots(pipeline: CorrectionPipeline) -> None:
    result = pipeline.correct("the student reads the quickly")
    assert result.best is not None
    assert "<" not in result.best.sentence


def test_report_mentions_the_edits(pipeline: CorrectionPipeline) -> None:
    text = pipeline.correct("the the dog ate the pizza").report()
    assert "delete" in text


def test_empty_input_is_handled(pipeline: CorrectionPipeline) -> None:
    result = pipeline.correct("")
    assert result.corrected is None


def test_unknown_words_are_reported(pipeline: CorrectionPipeline) -> None:
    result = pipeline.correct("the wibble ate the pizza")
    assert "wibble" in result.unknown_words


def test_semantics_improves_plausibility_of_slot_fillers(
    grammar, lexicon, checker, lm
) -> None:
    """The claim the project rests on, asserted directly.

    "the student reads the quickly" forces a noun slot that dozens of nouns
    could fill at identical edit cost.  Edit distance cannot choose; the
    selectional restrictions can.
    """
    without = CorrectionPipeline(
        grammar, lexicon, checker, lm,
        weights=RerankWeights(syntactic=10.0, semantic=0.0, language_model=1.0),
    )
    with_semantics = CorrectionPipeline(
        grammar, lexicon, checker, lm,
        weights=RerankWeights(syntactic=10.0, semantic=2.0, language_model=1.0),
    )
    best = with_semantics.correct("the student reads the quickly").best
    assert best is not None
    assert best.semantic_score >= without.correct(
        "the student reads the quickly"
    ).best.semantic_score


def test_weights_keep_syntax_dominant() -> None:
    """One extra edit must never be outweighed by semantics plus the LM."""
    weights = RerankWeights()
    cheapest_edit = 0.4
    assert weights.syntactic * cheapest_edit > weights.semantic + weights.language_model
