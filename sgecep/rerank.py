"""Slot filling and reranking.

The parser hands over repair *templates* with unfilled slots.  This module
turns each template into concrete sentences and picks a winner.

Two stages:

**Filling.**  Each slot is a pre-terminal, so its candidate words are the
lexicon entries carrying that tag -- often dozens.  Enumerating every
combination is hopeless, so filling is greedy left-to-right under the
language model, with one linguistically motivated exception: an INFLECT slot
is restricted to words sharing the original token's lemma.  That restriction
is what makes agreement repair deterministic -- ``reads`` can only become
``read``, never ``write``.

**Reranking.**  Candidates are scored by

    total = w_syn * (-syntactic_cost)
          + w_sem * semantic_score
          + w_lm  * lm_score_normalised

Weights are deliberately ordered so that syntax dominates: a repair that is
one whole edit cheaper can never be overturned by semantics or the LM.  The
semantic term only ever decides between repairs the parser already considers
equally good, which is the right division of labour -- a selectional
restriction is a soft preference, not a grammaticality judgement, and letting
it override the parser would mean rejecting perfectly grammatical sentences
for being surprising.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ec_earley import Repair
from .edits import Edit, EditKind
from .grammar import Lexicon
from .lm import BigramLM
from .semantics import SelectionalChecker, SemanticReport
from .trees import Tree


@dataclass
class Candidate:
    """A fully specified correction, with every score that produced it."""

    tokens: list[str]
    tree: Tree
    edits: list[Edit]
    syntactic_cost: float
    semantic_score: float = 0.5
    semantic_checks: int = 0
    lm_score: float = 0.0
    total_score: float = 0.0
    semantic_report: SemanticReport | None = None

    @property
    def sentence(self) -> str:
        return " ".join(self.tokens)

    @property
    def edit_count(self) -> int:
        return sum(1 for e in self.edits if e.kind is not EditKind.MATCH)

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.sentence}  [cost {self.syntactic_cost:.2f}, "
            f"sem {self.semantic_score:.2f}, lm {self.lm_score:.2f}, "
            f"total {self.total_score:.3f}]"
        )


@dataclass(frozen=True)
class RerankWeights:
    """Relative influence of the three signals.

    ``syntactic`` is large because the minimum-edit-distance criterion is the
    parser's actual guarantee; the other two break ties inside it.  With these
    defaults a 0.4 cost difference (one re-inflection) outweighs the entire
    range of the semantic and LM terms combined, so the ordering above holds
    by construction rather than by luck.
    """

    syntactic: float = 10.0
    semantic: float = 2.0
    language_model: float = 1.0


class Reranker:
    """Fills slots and ranks the resulting candidate corrections."""

    def __init__(
        self,
        lexicon: Lexicon,
        checker: SelectionalChecker,
        lm: BigramLM | None = None,
        weights: RerankWeights | None = None,
        fillers_per_slot: int = 6,
    ) -> None:
        self.lexicon = lexicon
        self.checker = checker
        self.lm = lm
        self.weights = weights or RerankWeights()
        self.fillers_per_slot = fillers_per_slot

        self._lemma_index: dict[str, set[str]] = {}
        for entry in lexicon.entries:
            self._lemma_index.setdefault(entry.lemma, set()).add(entry.word)

    # -- slot filling -------------------------------------------------------

    def _candidate_words(self, slot: Tree, edit: Edit | None) -> list[str]:
        """Words that may fill ``slot``, most constrained first."""
        words = self.lexicon.words(slot.label)
        if not words:
            return []

        if edit is not None and edit.kind is EditKind.INFLECT and edit.original:
            # Agreement repair: stay inside the original word's paradigm.
            lemma = self.lexicon.lemma(edit.original)
            same_lemma = self._lemma_index.get(lemma, set()) & words
            if same_lemma:
                return sorted(same_lemma)
        return sorted(words)

    def fill(self, repair: Repair) -> Tree:
        """Fill every slot in ``repair``'s tree, greedily under the LM."""
        tree = repair.tree.copy()
        leaves = tree.leaves()

        # Map each slot to the edit that created it, matched in order so that
        # INFLECT slots keep their original token.
        pending = [
            e
            for e in repair.edits
            if e.kind in (EditKind.INFLECT, EditKind.SUBSTITUTE, EditKind.INSERT)
        ]
        slot_edits: dict[int, Edit] = {}
        cursor = 0
        for index, leaf in enumerate(leaves):
            if leaf.word is None and cursor < len(pending):
                slot_edits[index] = pending[cursor]
                cursor += 1

        for index, leaf in enumerate(leaves):
            if leaf.word is not None:
                continue
            options = self._candidate_words(leaf, slot_edits.get(index))
            if not options:
                leaf.word = f"<{leaf.label}>"
                continue
            if len(options) == 1 or self.lm is None:
                leaf.word = options[0]
                continue

            previous = leaves[index - 1].word if index > 0 else "<s>"
            following = leaves[index + 1].word if index + 1 < len(leaves) else None

            def local_score(word: str) -> float:
                score = self.lm.probability(word, previous or "<s>")
                if following is not None:
                    # Look one token ahead too: a determiner must fit the noun
                    # after it, not merely the word before it.
                    score *= self.lm.probability(following, word)
                return score

            leaf.word = max(options, key=local_score)
        return tree

    # -- scoring ------------------------------------------------------------

    def _attach_replacements(self, repair: Repair, tree: Tree) -> list[Edit]:
        """Record the chosen word on each edit, for the edit script."""
        leaves = tree.leaves()
        chosen = [
            leaf.word
            for leaf in leaves
            if leaf.origin in ("inserted", "substituted") and leaf.word is not None
        ]
        out: list[Edit] = []
        cursor = 0
        for edit in repair.edits:
            if edit.kind in (EditKind.INFLECT, EditKind.SUBSTITUTE, EditKind.INSERT):
                if cursor < len(chosen):
                    out.append(edit.with_replacement(chosen[cursor]))
                    cursor += 1
                    continue
            out.append(edit)
        return out

    def rank(self, repairs: list[Repair], top_k: int = 5) -> list[Candidate]:
        candidates: list[Candidate] = []
        for repair in repairs:
            tree = self.fill(repair)
            tokens = [w for w in tree.yield_words()]
            report = self.checker.score(tree)
            lm_score = self.lm.score(tokens) if self.lm is not None else 0.0
            candidates.append(
                Candidate(
                    tokens=tokens,
                    tree=tree,
                    edits=self._attach_replacements(repair, tree),
                    syntactic_cost=repair.syntactic_cost,
                    semantic_score=report.score,
                    semantic_checks=report.checks,
                    lm_score=lm_score,
                    semantic_report=report,
                )
            )

        if not candidates:
            return []

        # Normalise the LM term into [0, 1] across this candidate set, so the
        # weights mean the same thing regardless of corpus or sentence length.
        lm_values = [c.lm_score for c in candidates]
        lo, hi = min(lm_values), max(lm_values)
        span = hi - lo

        for candidate in candidates:
            lm_normalised = 0.5 if span < 1e-9 else (candidate.lm_score - lo) / span
            candidate.total_score = (
                self.weights.syntactic * (-candidate.syntactic_cost)
                + self.weights.semantic * candidate.semantic_score
                + self.weights.language_model * lm_normalised
            )

        candidates.sort(
            key=lambda c: (
                -c.total_score,
                c.syntactic_cost,
                -c.semantic_checks,
                c.edit_count,
            )
        )

        deduped: list[Candidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.sentence in seen:
                continue
            seen.add(candidate.sentence)
            deduped.append(candidate)
            if len(deduped) >= top_k:
                break
        return deduped
