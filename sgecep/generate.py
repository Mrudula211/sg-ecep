"""Random sentence generation from the CFG.

Used to build the language-model corpus in ``data/corpus.txt``: sample
derivations from the grammar, keep the ones the selectional-restriction
checker judges plausible, and train the n-gram model on the survivors.  This
keeps the language model consistent with the grammar it is reranking against
-- an LM trained on outside text would penalise perfectly grammatical
sentences simply for using this grammar's small vocabulary.

Generation also makes the parser's two directions visible side by side: the
same grammar that *recognises* a sentence here *produces* one, which is the
clearest way to show in the report that a CFG is a declarative object and the
parser is only one way to use it.
"""

from __future__ import annotations

import random

from .grammar import Grammar, Lexicon
from .trees import Tree


class GenerationError(RuntimeError):
    """Raised when sampling cannot finish within the depth budget."""


class Generator:
    """Samples derivations from a grammar top-down."""

    def __init__(
        self,
        grammar: Grammar,
        lexicon: Lexicon,
        seed: int | None = None,
        max_depth: int = 14,
    ) -> None:
        self.grammar = grammar
        self.lexicon = lexicon
        self.random = random.Random(seed)
        self.max_depth = max_depth

    def _expand(self, symbol: str, depth: int) -> Tree:
        if self.grammar.is_preterminal(symbol):
            words = sorted(self.lexicon.words(symbol))
            if not words:
                raise GenerationError(f"no lexical entries for {symbol!r}")
            return Tree(symbol, word=self.random.choice(words), terminal=True)

        rules = self.grammar.rules_for(symbol)
        if not rules:
            raise GenerationError(f"no rules for {symbol!r}")

        if depth >= self.max_depth:
            # Past the budget, prefer the shortest right-hand side so the
            # derivation terminates instead of recursing forever.
            shortest = min(len(rule.rhs) for rule in rules)
            rules = [rule for rule in rules if len(rule.rhs) == shortest]

        rule = self.random.choice(rules)
        children = [self._expand(sym, depth + 1) for sym in rule.rhs]
        return Tree(symbol, children=children)

    def sample_tree(self, attempts: int = 40) -> Tree:
        for _ in range(attempts):
            try:
                return self._expand(self.grammar.start, 0)
            except (GenerationError, RecursionError):
                continue
        raise GenerationError("could not sample a derivation within the budget")

    def sample(self, attempts: int = 40) -> list[str]:
        return self.sample_tree(attempts).yield_words()

    def sample_many(
        self,
        count: int,
        min_length: int = 3,
        max_length: int = 12,
        unique: bool = True,
    ) -> list[list[str]]:
        """Sample ``count`` sentences within a length band."""
        out: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        guard = 0
        while len(out) < count and guard < count * 400:
            guard += 1
            try:
                tokens = self.sample()
            except GenerationError:
                continue
            if not (min_length <= len(tokens) <= max_length):
                continue
            if unique:
                key = tuple(tokens)
                if key in seen:
                    continue
                seen.add(key)
            out.append(tokens)
        return out
