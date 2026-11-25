"""Baseline Earley parser.

This is the textbook algorithm (Earley 1970) with no error correction.  It
exists for two reasons: it is the control condition in the evaluation (how
many corrupted sentences can a *plain* parser handle?  none, by construction),
and its chart is what the report uses to explain PREDICT / SCAN / COMPLETE
before the error-correcting extension is introduced.

The implementation keeps the chart as a list of ordered sets of items so the
chart dump in the demo matches the hand-traced tables in the lecture notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .grammar import Grammar, Lexicon, Rule
from .trees import Tree


@dataclass(frozen=True)
class Item:
    """An Earley item: a dotted rule plus the input position it started at."""

    rule: Rule
    dot: int
    origin: int

    @property
    def is_complete(self) -> bool:
        return self.dot >= len(self.rule.rhs)

    @property
    def next_symbol(self) -> str | None:
        if self.is_complete:
            return None
        return self.rule.rhs[self.dot]

    def advanced(self) -> "Item":
        return Item(self.rule, self.dot + 1, self.origin)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.rule.dotted(self.dot)} @{self.origin}"


class EarleyParser:
    """Recognise and parse sentences with a plain Earley chart."""

    def __init__(self, grammar: Grammar, lexicon: Lexicon) -> None:
        self.grammar = grammar
        self.lexicon = lexicon

    # -- chart construction -------------------------------------------------

    def build_chart(self, tokens: list[str]) -> list[list[Item]]:
        n = len(tokens)
        chart: list[list[Item]] = [[] for _ in range(n + 1)]
        seen: list[set[Item]] = [set() for _ in range(n + 1)]

        def add(position: int, item: Item) -> None:
            if item not in seen[position]:
                seen[position].add(item)
                chart[position].append(item)

        for rule in self.grammar.rules_for(self.grammar.start):
            add(0, Item(rule, 0, 0))

        for position in range(n + 1):
            cursor = 0
            while cursor < len(chart[position]):
                item = chart[position][cursor]
                cursor += 1

                if item.is_complete:
                    # COMPLETE: advance every item waiting on this non-terminal.
                    for waiting in list(chart[item.origin]):
                        if waiting.next_symbol == item.rule.lhs:
                            add(position, waiting.advanced())
                    continue

                symbol = item.next_symbol
                assert symbol is not None

                if self.grammar.is_nonterminal(symbol):
                    # PREDICT
                    for rule in self.grammar.rules_for(symbol):
                        add(position, Item(rule, 0, position))
                    # Epsilon fix: a nullable prediction must be completed now.
                    if symbol in self.grammar.nullable():
                        add(position, item.advanced())
                elif position < n:
                    # SCAN
                    if symbol in self.lexicon.tags(tokens[position]):
                        add(position + 1, item.advanced())

        return chart

    # -- recognition --------------------------------------------------------

    def accepting_items(self, chart: list[list[Item]], n: int) -> list[Item]:
        return [
            item
            for item in chart[n]
            if item.is_complete
            and item.origin == 0
            and item.rule.lhs == self.grammar.start
        ]

    def recognise(self, tokens: list[str]) -> bool:
        chart = self.build_chart(tokens)
        return bool(self.accepting_items(chart, len(tokens)))

    # -- parse extraction ---------------------------------------------------

    def parse(self, tokens: list[str], limit: int = 20) -> list[Tree]:
        """Every parse of ``tokens``, up to ``limit`` trees.

        Extraction re-walks the chart top-down rather than storing
        backpointers.  That is slower than the usual shared-packed forest but
        it is far easier to read, and the error-correcting parser in
        ``ec_earley`` does keep explicit backpointers, so the contrast is
        instructive.
        """
        chart = self.build_chart(tokens)
        n = len(tokens)
        if not self.accepting_items(chart, n):
            return []

        completed: dict[tuple[str, int, int], list[Rule]] = {}
        for end, items in enumerate(chart):
            for item in items:
                if item.is_complete:
                    completed.setdefault((item.rule.lhs, item.origin, end), []).append(
                        item.rule
                    )

        results: list[Tree] = []
        for tree in self._build(self.grammar.start, 0, n, tokens, completed, set()):
            results.append(tree)
            if len(results) >= limit:
                break
        return results

    def _build(
        self,
        symbol: str,
        start: int,
        end: int,
        tokens: list[str],
        completed: dict[tuple[str, int, int], list[Rule]],
        active: set[tuple[str, int, int]],
    ) -> Iterator[Tree]:
        if self.grammar.is_preterminal(symbol):
            if end == start + 1 and symbol in self.lexicon.tags(tokens[start]):
                yield Tree(symbol, word=tokens[start])
            return

        key = (symbol, start, end)
        if key in active:  # cycle guard for recursive unit rules
            return
        active = active | {key}

        for rule in completed.get(key, []):
            for children in self._build_rhs(
                rule.rhs, 0, start, end, tokens, completed, active
            ):
                yield Tree(symbol, children=list(children))

    def _build_rhs(
        self,
        rhs: tuple[str, ...],
        index: int,
        start: int,
        end: int,
        tokens: list[str],
        completed: dict[tuple[str, int, int], list[Rule]],
        active: set[tuple[str, int, int]],
    ) -> Iterator[list[Tree]]:
        if index == len(rhs):
            if start == end:
                yield []
            return

        symbol = rhs[index]
        is_last = index == len(rhs) - 1
        for split in range(start, end + 1):
            if is_last and split != end:
                continue
            for subtree in self._build(symbol, start, split, tokens, completed, active):
                for rest in self._build_rhs(
                    rhs, index + 1, split, end, tokens, completed, active
                ):
                    yield [subtree] + rest

    # -- diagnostics --------------------------------------------------------

    def dump_chart(self, tokens: list[str], max_items: int = 25) -> str:
        """Human-readable chart, for the report and the ``--chart`` flag."""
        chart = self.build_chart(tokens)
        lines: list[str] = []
        for position, items in enumerate(chart):
            header = f"--- Chart[{position}]"
            if position < len(tokens):
                header += f"   (next token: {tokens[position]!r})"
            lines.append(header)
            for item in items[:max_items]:
                lines.append(f"    {item}")
            if len(items) > max_items:
                lines.append(f"    ... {len(items) - max_items} more items")
        return "\n".join(lines)
