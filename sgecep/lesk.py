"""Word sense disambiguation by the Lesk algorithm.

Two variants are implemented, both from the lecture material:

*Simplified Lesk* scores each candidate sense of the target word by the
overlap between its signature (gloss + examples) and the surrounding context
words.

*Extended Lesk* widens the signature with the glosses of related synsets
(hypernyms, hyponyms, meronyms).  WordNet glosses are short, so plain Lesk
very often produces an all-zero tie; the extension is what makes the
disambiguation actually discriminate in practice.

Both deliberately return a ranked list with scores rather than a single
synset, because the reranker needs to know *how confident* the
disambiguation was: a sense chosen on a one-word overlap should not be
allowed to veto a syntactic repair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

STOPWORDS = frozenset(
    """a an the this that these those of in on at to from with for by and or but
    is are was were be been being am do does did have has had will would can could
    shall should may might must it its as not no so if then than there their his
    her him she he they them we us you your my me i which who whom what when where
    how all any some each every both few more most other such only own same too very
    one two three s t don now d ll m o re ve y""".split()
)

_TOKEN_RE = re.compile(r"[a-z]+")


def bag_of_words(text: str) -> set[str]:
    """Lower-cased content words of ``text``, stopwords removed."""
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 2
    }


@dataclass(frozen=True)
class SenseScore:
    """One candidate sense and the evidence for it."""

    synset_name: str
    definition: str
    score: float                 # final ranking score (overlap + sense prior)
    overlap: frozenset[str]
    lesk_overlap: int = 0        # raw gloss overlap, before the prior
    sense_rank: int = 0          # 0-based position in WordNet's sense order

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.synset_name} (overlap {self.lesk_overlap}, "
            f"rank {self.sense_rank}) :: {self.definition}"
        )


class LeskDisambiguator:
    """Lesk WSD over WordNet, with graceful degradation when it is absent."""

    def __init__(self, extended: bool = True, mfs_weight: float = 1.5) -> None:
        """``mfs_weight`` interpolates WordNet's most-frequent-sense prior.

        Plain Lesk is well known to score *below* the most-frequent-sense
        baseline on all-words WSD: glosses are short, so overlaps are sparse
        and a single coincidental word can promote a rare sense.  Scoring
        ``overlap + mfs_weight / (rank + 1)`` keeps Lesk in charge whenever it
        has real evidence -- a rare sense needs roughly two more overlapping
        words than sense 1 to win -- while falling back on frequency when it
        has none.  Set to 0.0 to recover textbook Lesk.
        """
        self.extended = extended
        self.mfs_weight = mfs_weight
        self._wn = None
        self._available: bool | None = None

    # -- WordNet access -----------------------------------------------------

    @property
    def available(self) -> bool:
        """True when WordNet can be loaded; cached after the first attempt."""
        if self._available is None:
            try:
                from nltk.corpus import wordnet as wn

                wn.synsets("dog", "n")  # force the lazy corpus loader
                self._wn = wn
                self._available = True
            except Exception:
                self._wn = None
                self._available = False
        return self._available

    @property
    def wn(self):
        if not self.available:
            raise RuntimeError("WordNet is not available")
        return self._wn

    # -- signatures ---------------------------------------------------------

    def signature(self, synset) -> set[str]:
        words = bag_of_words(synset.definition())
        for example in synset.examples():
            words |= bag_of_words(example)
        for lemma in synset.lemmas():
            words.add(lemma.name().lower().replace("_", " "))

        if self.extended:
            related = []
            related.extend(synset.hypernyms())
            related.extend(synset.hyponyms())
            related.extend(synset.part_meronyms())
            related.extend(synset.member_holonyms())
            for other in related[:12]:  # cap: glosses get noisy fast
                words |= bag_of_words(other.definition())
        return words

    # -- disambiguation -----------------------------------------------------

    def rank_senses(
        self, word: str, context: list[str], pos: str | None = "n"
    ) -> list[SenseScore]:
        """Rank the senses of ``word`` given ``context``, best first."""
        if not self.available:
            return []
        synsets = self.wn.synsets(word.lower(), pos) if pos else self.wn.synsets(word.lower())
        if not synsets:
            return []

        context_bag: set[str] = set()
        for other in context:
            if other.lower() == word.lower():
                continue
            context_bag |= bag_of_words(other)
            # Pull in the glosses of context words too (Banerjee & Pedersen's
            # extension); without it the context bag is just a few bare tokens.
            if self.extended:
                for sense in self.wn.synsets(other.lower())[:3]:
                    context_bag |= bag_of_words(sense.definition())

        scored: list[SenseScore] = []
        for rank, synset in enumerate(synsets):
            overlap = self.signature(synset) & context_bag
            prior = self.mfs_weight / (rank + 1.0)
            scored.append(
                SenseScore(
                    synset_name=synset.name(),
                    definition=synset.definition(),
                    score=len(overlap) + prior,
                    overlap=frozenset(overlap),
                    lesk_overlap=len(overlap),
                    sense_rank=rank,
                )
            )
        scored.sort(key=lambda s: (-s.score, s.sense_rank))
        return scored

    def disambiguate(
        self, word: str, context: list[str], pos: str | None = "n"
    ) -> SenseScore | None:
        ranked = self.rank_senses(word, context, pos)
        return ranked[0] if ranked else None

    @lru_cache(maxsize=4096)
    def hypernym_closure(self, synset_name: str) -> frozenset[str]:
        """Every synset on any hypernym path from ``synset_name`` upward."""
        if not self.available:
            return frozenset()
        try:
            synset = self.wn.synset(synset_name)
        except Exception:
            return frozenset()
        names = {synset.name()}
        for path in synset.hypernym_paths():
            names.update(node.name() for node in path)
        return frozenset(names)
