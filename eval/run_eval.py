"""Evaluation harness.

Measures how often the system recovers the original sentence from a corrupted
one, and -- the result the project actually turns on -- how much of that
recovery is due to the semantic layer rather than the parser alone.

Four configurations are compared:

``baseline``       plain Earley.  Cannot repair anything, so its recovery rate
                   is zero by construction.  Reported to make the point that
                   the comparison is against *nothing*, not against a weaker
                   corrector.
``syntax-only``    error-correcting Earley, cheapest repair, ties broken
                   arbitrarily (first in enumeration order).
``syntax+lm``      ties broken by the language model.
``full``           ties broken by semantics and then the language model.

A third metric, *semantic validity*, reports the share of top-ranked
corrections that satisfy every applicable selectional restriction.  Exact
recovery alone understates the semantic layer: when a slot admits sixty nouns,
insisting on the one the gold sentence happened to use penalises a correction
that is perfectly sensible but different.  Semantic validity asks the question
the layer is actually responsible for -- is the correction *plausible*?

Two metrics are reported.  *Exact recovery* is the share of examples whose
top-ranked correction is exactly the gold sentence.  *Oracle recovery* is the
share whose candidate list contains the gold sentence anywhere -- the ceiling
the reranker is working against, and the right way to separate "the parser
never found it" from "the reranker ranked it second".

Both are computed over the *genuinely ungrammatical* examples only.  This
matters, and it is easy to get wrong: corrupting a sentence often yields
another perfectly grammatical sentence -- swapping one noun for another almost
always does, and dropping an adjective or an adverb usually does.  Scoring
those as failures would be measuring the wrong thing, because the system is
right to leave a grammatical sentence alone.  They are counted and reported
separately as "silent" corruptions.

Run with::

    python -m eval.run_eval --limit 150
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

from eval.corrupt import CorruptedExample, Corruptor, corrupt_corpus
from sgecep.grammar import Grammar, Lexicon
from sgecep.lesk import LeskDisambiguator
from sgecep.lm import BigramLM
from sgecep.pipeline import CorrectionPipeline
from sgecep.rerank import RerankWeights
from sgecep.semantics import SelectionalChecker

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Outcome:
    name: str
    total: int = 0
    exact: int = 0
    oracle: int = 0
    valid: int = 0
    no_repair: int = 0
    seconds: float = 0.0
    by_kind: dict[str, list[int]] = field(default_factory=dict)

    def record(
        self,
        kind: str,
        hit: bool,
        in_list: bool,
        found: bool,
        plausible: bool = False,
    ) -> None:
        self.total += 1
        self.exact += int(hit)
        self.oracle += int(in_list)
        self.valid += int(plausible)
        self.no_repair += int(not found)
        bucket = self.by_kind.setdefault(kind, [0, 0])
        bucket[0] += 1
        bucket[1] += int(hit)

    @property
    def exact_rate(self) -> float:
        return self.exact / self.total if self.total else 0.0

    @property
    def oracle_rate(self) -> float:
        return self.oracle / self.total if self.total else 0.0

    @property
    def valid_rate(self) -> float:
        return self.valid / self.total if self.total else 0.0

    def row(self) -> str:
        return (
            f"{self.name:<14} {self.total:>6} {self.exact_rate:>10.1%} "
            f"{self.oracle_rate:>10.1%} {self.valid_rate:>10.1%} "
            f"{self.no_repair:>10} {self.seconds:>8.1f}s"
        )


def _load_corpus(path: Path, limit: int | None) -> list[list[str]]:
    sentences = [
        line.split()
        for raw in path.read_text(encoding="utf-8").splitlines()
        if (line := raw.split("#", 1)[0].strip())
    ]
    return sentences[:limit] if limit else sentences


def _build(
    grammar: Grammar,
    lexicon: Lexicon,
    checker: SelectionalChecker,
    lm: BigramLM | None,
    weights: RerankWeights,
    max_cost: float,
) -> CorrectionPipeline:
    return CorrectionPipeline(
        grammar, lexicon, checker, lm, weights=weights, max_cost=max_cost
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=150, help="gold sentences to use")
    parser.add_argument("--max-cost", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-wordnet", action="store_true")
    args = parser.parse_args()

    grammar = Grammar.from_file(DATA / "grammar.cfg")
    lexicon = Lexicon.from_file(DATA / "lexicon.tsv")
    disambiguator = _Offline() if args.no_wordnet else LeskDisambiguator()
    checker = SelectionalChecker.from_files(
        DATA / "selectional_restrictions.json",
        DATA / "semantic_types.json",
        lexicon=lexicon,
        disambiguator=disambiguator,
    )
    lm = BigramLM.from_corpus_file(DATA / "corpus.txt")

    gold = _load_corpus(DATA / "corpus.txt", args.limit)
    all_examples = corrupt_corpus(gold, lexicon, seed=args.seed)

    # Split off corruptions that left the sentence grammatical: the system is
    # correct to return those unchanged, so including them would understate
    # the corrector and overstate nothing useful.
    probe = Grammar.from_file(DATA / "grammar.cfg")
    from sgecep.earley import EarleyParser

    recogniser = EarleyParser(probe, lexicon)
    examples = [e for e in all_examples if not recogniser.recognise(e.corrupted)]
    silent = len(all_examples) - len(examples)

    print(f"gold sentences : {len(gold)}")
    print(f"corruptions    : {len(all_examples)}")
    print(
        f"  ungrammatical: {len(examples)}  (scored)\n"
        f"  still grammatical: {silent}  "
        f"({silent / max(len(all_examples), 1):.1%}, left unchanged by design)"
    )
    print(f"WordNet backend: {checker.types.uses_wordnet}")
    print(f"language model : {lm}")
    print()

    configurations = {
        "syntax-only": _build(
            grammar, lexicon, checker, None,
            RerankWeights(syntactic=10.0, semantic=0.0, language_model=0.0),
            args.max_cost,
        ),
        "syntax+lm": _build(
            grammar, lexicon, checker, lm,
            RerankWeights(syntactic=10.0, semantic=0.0, language_model=1.0),
            args.max_cost,
        ),
        "full": _build(
            grammar, lexicon, checker, lm,
            RerankWeights(syntactic=10.0, semantic=2.0, language_model=1.0),
            args.max_cost,
        ),
    }

    # The baseline is a plain recogniser.  Every scored example is
    # ungrammatical, so it repairs none of them -- that is the point.
    baseline = Outcome("baseline")
    for example in examples:
        baseline.record(example.corruption, hit=False, in_list=False, found=False)

    outcomes = [baseline]
    for name, pipeline in configurations.items():
        outcome = Outcome(name)
        started = time.time()
        for example in examples:
            result = pipeline.correct(example.corrupted, top_k=args.top_k)
            sentences = [c.sentence for c in result.candidates]
            target = " ".join(example.gold)
            top = result.candidates[0] if result.candidates else None
            outcome.record(
                example.corruption,
                hit=bool(sentences) and sentences[0] == target,
                in_list=target in sentences,
                found=bool(sentences),
                plausible=bool(top is not None and top.semantic_score >= 0.999),
            )
        outcome.seconds = time.time() - started
        outcomes.append(outcome)

    header = (
        f"{'configuration':<14} {'cases':>6} {'exact':>10} "
        f"{'oracle':>10} {'valid':>10} {'no repair':>10} {'time':>9}"
    )
    print(header)
    print("-" * len(header))
    for outcome in outcomes:
        print(outcome.row())

    print("\nexact recovery by corruption type")
    kinds = sorted(Corruptor.KINDS)
    print(f"{'configuration':<14} " + " ".join(f"{k:>12}" for k in kinds))
    print("-" * (14 + 13 * len(kinds)))
    for outcome in outcomes:
        cells = []
        for kind in kinds:
            total, hits = outcome.by_kind.get(kind, [0, 0])
            cells.append(f"{(hits / total if total else 0):>12.1%}")
        print(f"{outcome.name:<14} " + " ".join(cells))

    ambiguous = _tie_analysis(configurations["full"], examples, args.top_k)
    print(
        f"\nties: in {ambiguous[0]} of {len(examples)} cases the parser returned "
        f"more than one cheapest repair;\n"
        f"      the full configuration recovered the gold sentence in "
        f"{ambiguous[1]} of those."
    )


def _tie_analysis(
    pipeline: CorrectionPipeline, examples: list[CorruptedExample], top_k: int
) -> tuple[int, int]:
    """Count cases with several minimum-cost repairs -- where reranking matters."""
    tied = 0
    recovered = 0
    for example in examples:
        result = pipeline.correct(example.corrupted, top_k=top_k)
        if not result.candidates:
            continue
        cheapest = min(c.syntactic_cost for c in result.candidates)
        at_minimum = [
            c for c in result.candidates if abs(c.syntactic_cost - cheapest) < 1e-9
        ]
        if len(at_minimum) > 1:
            tied += 1
            if result.candidates[0].sentence == " ".join(example.gold):
                recovered += 1
    return tied, recovered


class _Offline(LeskDisambiguator):
    @property
    def available(self) -> bool:  # type: ignore[override]
        return False


if __name__ == "__main__":
    main()
