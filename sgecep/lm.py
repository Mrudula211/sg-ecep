"""Bigram language model with interpolated Kneser-Ney smoothing.

The LM is the third signal in the reranker.  Its job is narrow but real: when
two repairs are equally cheap syntactically *and* equally plausible
semantically, the LM decides which word actually belongs in a slot.  Nothing
in the selectional restrictions distinguishes ``a book`` from ``an book``, for
instance; the LM does.

Kneser-Ney rather than add-k because the discriminating case is exactly the
one add-k handles worst.  Add-k assigns an unseen bigram a probability
proportional to the unigram frequency of its second word, so a common word in
an impossible context still scores well.  Kneser-Ney replaces that unigram
term with a *continuation* probability -- how many distinct contexts a word
has been seen in -- which is what makes it prefer the word that fits the
context over the word that is merely frequent.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


class BigramLM:
    """Interpolated Kneser-Ney bigram model."""

    def __init__(self, discount: float = 0.75) -> None:
        if not 0.0 < discount < 1.0:
            raise ValueError("discount must lie strictly between 0 and 1")
        self.discount = discount

        self.unigrams: Counter[str] = Counter()
        self.bigrams: Counter[tuple[str, str]] = Counter()
        self.context_totals: Counter[str] = Counter()
        self.followers: dict[str, set[str]] = defaultdict(set)
        self.preceders: dict[str, set[str]] = defaultdict(set)
        self.distinct_bigrams: int = 0
        self.vocabulary: set[str] = set()
        self.trained: bool = False

    # -- training -----------------------------------------------------------

    def train(self, sentences: list[list[str]]) -> "BigramLM":
        for tokens in sentences:
            padded = [BOS] + [t.lower() for t in tokens] + [EOS]
            for token in padded:
                self.unigrams[token] += 1
                self.vocabulary.add(token)
            for left, right in zip(padded, padded[1:]):
                self.bigrams[(left, right)] += 1
                self.context_totals[left] += 1
                self.followers[left].add(right)
                self.preceders[right].add(left)
        self.distinct_bigrams = len(self.bigrams)
        self.vocabulary.add(UNK)
        self.trained = True
        return self

    @classmethod
    def from_corpus_file(cls, path: str | Path, discount: float = 0.75) -> "BigramLM":
        sentences = [
            line.split()
            for raw in Path(path).read_text(encoding="utf-8").splitlines()
            if (line := raw.split("#", 1)[0].strip())
        ]
        return cls(discount).train(sentences)

    # -- probability ---------------------------------------------------------

    def _normalise(self, token: str) -> str:
        token = token.lower()
        return token if token in self.vocabulary else UNK

    def continuation_probability(self, word: str) -> float:
        """P_cont(w): share of distinct bigram types that end in ``w``.

        This is the Kneser-Ney lower-order term.  A word seen in many
        different contexts gets a high value even if it is individually rare.
        """
        if self.distinct_bigrams == 0:
            return 1.0 / max(len(self.vocabulary), 1)
        contexts = len(self.preceders.get(word, ()))
        if contexts == 0:
            # Unseen word: one notional context, so the model stays proper.
            return 1.0 / (self.distinct_bigrams + len(self.vocabulary))
        return contexts / self.distinct_bigrams

    def probability(self, word: str, context: str) -> float:
        if not self.trained:
            raise RuntimeError("language model has not been trained")
        word = self._normalise(word)
        context = self._normalise(context)

        total = self.context_totals.get(context, 0)
        continuation = self.continuation_probability(word)
        if total == 0:
            return max(continuation, 1e-12)

        count = self.bigrams.get((context, word), 0)
        discounted = max(count - self.discount, 0.0) / total
        # lambda: the mass the discount freed up, redistributed by P_cont.
        lam = self.discount * len(self.followers.get(context, ())) / total
        return max(discounted + lam * continuation, 1e-12)

    def log_probability(self, tokens: list[str]) -> float:
        padded = [BOS] + [t.lower() for t in tokens] + [EOS]
        return sum(
            math.log(self.probability(right, left))
            for left, right in zip(padded, padded[1:])
        )

    def score(self, tokens: list[str]) -> float:
        """Mean log probability per token: length-independent, so repairs of
        different lengths can be compared."""
        if not tokens:
            return -1e9
        return self.log_probability(tokens) / (len(tokens) + 1)

    def perplexity(self, sentences: list[list[str]]) -> float:
        total_logprob = 0.0
        total_tokens = 0
        for tokens in sentences:
            total_logprob += self.log_probability(tokens)
            total_tokens += len(tokens) + 1
        if total_tokens == 0:
            return float("inf")
        return math.exp(-total_logprob / total_tokens)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"BigramLM(vocab={len(self.vocabulary)}, "
            f"bigram_types={self.distinct_bigrams}, discount={self.discount})"
        )


class AddKLM(BigramLM):
    """Add-k smoothed bigram model, kept as the evaluation baseline.

    Included so the report can quantify the Kneser-Ney claim above rather
    than assert it.
    """

    def __init__(self, k: float = 0.1) -> None:
        super().__init__(discount=0.75)
        self.k = k

    def probability(self, word: str, context: str) -> float:
        if not self.trained:
            raise RuntimeError("language model has not been trained")
        word = self._normalise(word)
        context = self._normalise(context)
        vocab = max(len(self.vocabulary), 1)
        numerator = self.bigrams.get((context, word), 0) + self.k
        denominator = self.context_totals.get(context, 0) + self.k * vocab
        return numerator / denominator
