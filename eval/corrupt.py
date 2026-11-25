"""Synthetic error injection.

Evaluation needs ungrammatical sentences whose correct form is known.  Rather
than hand-annotate a learner corpus, grammatical sentences are corrupted on
purpose and the system is measured on how often it recovers the original.

The four corruption types mirror the four edit operations, which makes the
evaluation a direct test of the cost model:

``agreement``   swap a verb or noun for another inflection of the same lemma
``drop_word``   delete a token (usually a determiner)
``duplicate``   repeat a token
``swap_word``   replace a token with a different word of the same tag
``swap_pos``    replace a token with a word carrying a DIFFERENT tag

``swap_pos`` is the corruption that tests the project's actual claim.
Replacing a noun with an adverb forces the parser to substitute a slot back in
that position, and that slot can be filled by any of sixty-odd nouns.  Minimum
edit distance cannot choose between them -- every filler costs the same -- so
the choice falls entirely to the semantic layer and the language model.  The
morphosyntactic corruptions, by contrast, almost always reduce to picking an
inflection, which the language model handles on its own.

One caveat the evaluation has to respect: corruption is not guaranteed to
produce an ungrammatical sentence, and it is not guaranteed that the original
is the *only* cheapest repair.  ``corrupt_corpus`` therefore records both the
gold sentence and whether the corrupted form is genuinely ungrammatical, so
the harness can report recovery over the cases that actually posed a problem.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sgecep.grammar import Lexicon

AGREEMENT_SENSITIVE = ("NNS", "NNP")


@dataclass
class CorruptedExample:
    gold: list[str]
    corrupted: list[str]
    corruption: str
    position: int

    @property
    def changed(self) -> bool:
        return self.gold != self.corrupted


class Corruptor:
    """Injects a single error into a grammatical sentence."""

    def __init__(self, lexicon: Lexicon, seed: int | None = None) -> None:
        self.lexicon = lexicon
        self.random = random.Random(seed)

        self._lemma_forms: dict[str, set[str]] = {}
        for entry in lexicon.entries:
            self._lemma_forms.setdefault(entry.lemma, set()).add(entry.word)

    # -- individual corruptions ---------------------------------------------

    def _agreement(self, tokens: list[str]) -> tuple[list[str], int] | None:
        """Swap a token for a different inflection of the same lemma."""
        order = list(range(len(tokens)))
        self.random.shuffle(order)
        for index in order:
            word = tokens[index]
            lemma = self.lexicon.lemma(word)
            alternatives = sorted(self._lemma_forms.get(lemma, set()) - {word})
            if alternatives:
                out = list(tokens)
                out[index] = self.random.choice(alternatives)
                return out, index
        return None

    def _drop_word(self, tokens: list[str]) -> tuple[list[str], int] | None:
        if len(tokens) < 3:
            return None
        index = self.random.randrange(len(tokens))
        return tokens[:index] + tokens[index + 1 :], index

    def _duplicate(self, tokens: list[str]) -> tuple[list[str], int] | None:
        index = self.random.randrange(len(tokens))
        return tokens[: index + 1] + [tokens[index]] + tokens[index + 1 :], index

    def _swap_word(self, tokens: list[str]) -> tuple[list[str], int] | None:
        order = list(range(len(tokens)))
        self.random.shuffle(order)
        for index in order:
            tags = self.lexicon.tags(tokens[index])
            if not tags:
                continue
            tag = self.random.choice(sorted(tags))
            options = sorted(self.lexicon.words(tag) - {tokens[index]})
            if options:
                out = list(tokens)
                out[index] = self.random.choice(options)
                return out, index
        return None

    def _swap_pos(self, tokens: list[str]) -> tuple[list[str], int] | None:
        """Replace a token with a word of a different part of speech."""
        order = list(range(len(tokens)))
        self.random.shuffle(order)
        for index in order:
            own_tags = self.lexicon.tags(tokens[index])
            if not own_tags:
                continue
            options = sorted(
                word
                for word in self.lexicon.vocabulary
                if word != tokens[index] and not (self.lexicon.tags(word) & own_tags)
            )
            if options:
                out = list(tokens)
                out[index] = self.random.choice(options)
                return out, index
        return None

    # -- driver --------------------------------------------------------------

    KINDS = ("agreement", "drop_word", "duplicate", "swap_word", "swap_pos")

    def corrupt(
        self, tokens: list[str], kind: str | None = None
    ) -> CorruptedExample | None:
        kind = kind or self.random.choice(self.KINDS)
        handler = {
            "agreement": self._agreement,
            "drop_word": self._drop_word,
            "duplicate": self._duplicate,
            "swap_word": self._swap_word,
            "swap_pos": self._swap_pos,
        }[kind]
        outcome = handler(list(tokens))
        if outcome is None:
            return None
        corrupted, index = outcome
        return CorruptedExample(
            gold=list(tokens), corrupted=corrupted, corruption=kind, position=index
        )


def corrupt_corpus(
    sentences: list[list[str]],
    lexicon: Lexicon,
    seed: int = 7,
    kinds: tuple[str, ...] | None = None,
) -> list[CorruptedExample]:
    """One corrupted example per sentence per corruption kind."""
    corruptor = Corruptor(lexicon, seed=seed)
    kinds = kinds or Corruptor.KINDS
    out: list[CorruptedExample] = []
    for tokens in sentences:
        for kind in kinds:
            example = corruptor.corrupt(tokens, kind)
            if example is not None and example.changed:
                out.append(example)
    return out
