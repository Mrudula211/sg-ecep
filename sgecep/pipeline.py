"""End-to-end pipeline: ungrammatical text in, corrected sentence out."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .earley import EarleyParser
from .ec_earley import ErrorCorrectingEarleyParser, Repair
from .edits import CostModel, EditKind, summarise
from .grammar import Grammar, Lexicon, check_coverage
from .lesk import LeskDisambiguator
from .lm import BigramLM
from .rerank import Candidate, RerankWeights, Reranker
from .semantics import SelectionalChecker

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenise(text: str) -> list[str]:
    """Lower-case word tokens; punctuation is dropped (the grammar has none)."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class CorrectionResult:
    """Everything the pipeline learned about one input sentence."""

    original: list[str]
    already_grammatical: bool
    candidates: list[Candidate] = field(default_factory=list)
    repairs: list[Repair] = field(default_factory=list)
    unknown_words: list[str] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def corrected(self) -> str | None:
        return self.best.sentence if self.best else None

    def report(self, verbose: bool = False, max_candidates: int = 3) -> str:
        lines = [f"input      : {' '.join(self.original)}"]
        if self.unknown_words:
            lines.append(f"unknown    : {', '.join(self.unknown_words)}")

        if self.already_grammatical:
            lines.append("status     : already grammatical")
        elif not self.candidates:
            lines.append("status     : no repair found within the cost budget")
        else:
            best = self.best
            assert best is not None
            lines.append(f"corrected  : {best.sentence}")
            lines.append(f"edits      : {summarise(best.edits)}")
            for edit in best.edits:
                if edit.kind is not EditKind.MATCH:
                    lines.append(f"             - {edit.describe()}")

        if self.candidates:
            best = self.candidates[0]
            lines.append(
                f"scores     : cost {best.syntactic_cost:.2f} | "
                f"semantic {best.semantic_score:.2f} "
                f"({best.semantic_checks} checks) | "
                f"lm {best.lm_score:.3f}"
            )

        if verbose and self.candidates:
            lines.append("candidates :")
            for rank, candidate in enumerate(self.candidates[:max_candidates], 1):
                lines.append(
                    f"  {rank}. {candidate.sentence}"
                    f"   [cost {candidate.syntactic_cost:.2f}, "
                    f"sem {candidate.semantic_score:.2f}, "
                    f"lm {candidate.lm_score:.3f}, "
                    f"total {candidate.total_score:.3f}]"
                )
            best = self.best
            if best is not None and best.semantic_report is not None:
                lines.append("semantics  :")
                for line in best.semantic_report.explain().splitlines():
                    lines.append(f"  {line}")
                lines.append("parse tree :")
                for line in best.tree.pretty().splitlines():
                    lines.append(f"  {line}")
        return "\n".join(lines)


class CorrectionPipeline:
    """Wires the grammar, parsers, semantics and LM into one object."""

    def __init__(
        self,
        grammar: Grammar,
        lexicon: Lexicon,
        checker: SelectionalChecker,
        lm: BigramLM | None = None,
        costs: CostModel | None = None,
        weights: RerankWeights | None = None,
        max_cost: float = 3.0,
        beam: int = 24,
    ) -> None:
        self.grammar = grammar
        self.lexicon = lexicon
        self.baseline = EarleyParser(grammar, lexicon)
        self.parser = ErrorCorrectingEarleyParser(
            grammar, lexicon, costs=costs, max_cost=max_cost, beam=beam
        )
        self.checker = checker
        self.lm = lm
        self.reranker = Reranker(lexicon, checker, lm, weights)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path = DATA_DIR,
        use_wordnet: bool = True,
        train_lm: bool = True,
        **kwargs,
    ) -> "CorrectionPipeline":
        data_dir = Path(data_dir)
        grammar = Grammar.from_file(data_dir / "grammar.cfg")
        lexicon = Lexicon.from_file(data_dir / "lexicon.tsv")

        problems = check_coverage(grammar, lexicon)
        if problems:
            raise ValueError(
                "grammar/lexicon mismatch:\n  " + "\n  ".join(problems)
            )

        disambiguator = LeskDisambiguator() if use_wordnet else _NoWordNet()
        checker = SelectionalChecker.from_files(
            data_dir / "selectional_restrictions.json",
            data_dir / "semantic_types.json",
            lexicon=lexicon,
            disambiguator=disambiguator,
        )

        lm = None
        corpus = data_dir / "corpus.txt"
        if train_lm and corpus.exists():
            lm = BigramLM.from_corpus_file(corpus)

        return cls(grammar, lexicon, checker, lm, **kwargs)

    # -- use ----------------------------------------------------------------

    def correct(
        self, text: str | list[str], top_k: int = 5, k_repairs: int = 10
    ) -> CorrectionResult:
        tokens = tokenise(text) if isinstance(text, str) else [t.lower() for t in text]
        unknown = [t for t in tokens if not self.lexicon.knows(t)]

        if not tokens:
            return CorrectionResult(original=[], already_grammatical=False)

        if self.baseline.recognise(tokens):
            trees = self.baseline.parse(tokens, limit=k_repairs)
            repairs = [Repair(tree=t, edits=[], syntactic_cost=0.0) for t in trees]
            candidates = self.reranker.rank(repairs, top_k=top_k)
            return CorrectionResult(
                original=tokens,
                already_grammatical=True,
                candidates=candidates,
                repairs=repairs,
                unknown_words=unknown,
            )

        repairs = self.parser.repairs(tokens, k=k_repairs)
        candidates = self.reranker.rank(repairs, top_k=top_k)
        return CorrectionResult(
            original=tokens,
            already_grammatical=False,
            candidates=candidates,
            repairs=repairs,
            unknown_words=unknown,
        )

    def is_grammatical(self, text: str | list[str]) -> bool:
        tokens = tokenise(text) if isinstance(text, str) else [t.lower() for t in text]
        return self.baseline.recognise(tokens)


class _NoWordNet(LeskDisambiguator):
    """Forces the bundled-ontology backend, for tests and offline runs."""

    @property
    def available(self) -> bool:  # type: ignore[override]
        return False
