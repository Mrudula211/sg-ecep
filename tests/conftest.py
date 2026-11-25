"""Shared fixtures.

The semantic fixtures default to the bundled ontology rather than WordNet, so
the suite runs identically whether or not the WordNet corpus is installed.
The WordNet path is exercised by its own test, which skips when absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sgecep.earley import EarleyParser
from sgecep.ec_earley import ErrorCorrectingEarleyParser
from sgecep.grammar import Grammar, Lexicon
from sgecep.lesk import LeskDisambiguator
from sgecep.lm import BigramLM
from sgecep.pipeline import CorrectionPipeline
from sgecep.semantics import SelectionalChecker

DATA = Path(__file__).resolve().parent.parent / "data"


def tokens(sentence: str) -> list[str]:
    return sentence.lower().split()


class OfflineDisambiguator(LeskDisambiguator):
    """Forces the bundled-ontology backend."""

    @property
    def available(self) -> bool:  # type: ignore[override]
        return False


@pytest.fixture(scope="session")
def grammar() -> Grammar:
    return Grammar.from_file(DATA / "grammar.cfg")


@pytest.fixture(scope="session")
def lexicon() -> Lexicon:
    return Lexicon.from_file(DATA / "lexicon.tsv")


@pytest.fixture(scope="session")
def parser(grammar: Grammar, lexicon: Lexicon) -> EarleyParser:
    return EarleyParser(grammar, lexicon)


@pytest.fixture(scope="session")
def ec_parser(grammar: Grammar, lexicon: Lexicon) -> ErrorCorrectingEarleyParser:
    return ErrorCorrectingEarleyParser(grammar, lexicon, max_cost=3.0)


@pytest.fixture(scope="session")
def checker(lexicon: Lexicon) -> SelectionalChecker:
    return SelectionalChecker.from_files(
        DATA / "selectional_restrictions.json",
        DATA / "semantic_types.json",
        lexicon=lexicon,
        disambiguator=OfflineDisambiguator(),
    )


@pytest.fixture(scope="session")
def lm() -> BigramLM:
    return BigramLM.from_corpus_file(DATA / "corpus.txt")


@pytest.fixture(scope="session")
def pipeline(
    grammar: Grammar,
    lexicon: Lexicon,
    checker: SelectionalChecker,
    lm: BigramLM,
) -> CorrectionPipeline:
    return CorrectionPipeline(grammar, lexicon, checker, lm, max_cost=3.0)


@pytest.fixture(scope="session")
def wordnet_checker(lexicon: Lexicon) -> SelectionalChecker:
    checker = SelectionalChecker.from_files(
        DATA / "selectional_restrictions.json",
        DATA / "semantic_types.json",
        lexicon=lexicon,
    )
    if not checker.types.uses_wordnet:
        pytest.skip("WordNet is not installed")
    return checker
