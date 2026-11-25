"""Context-free grammar and lexicon representation.

The parser separates two layers:

``Grammar``
    Pure syntactic rules over non-terminals and pre-terminals (POS tags),
    loaded from ``data/grammar.cfg``.

``Lexicon``
    The word -> POS-tag mapping, loaded from ``data/lexicon.tsv``.  Keeping
    the lexicon out of the grammar matters for error correction: a
    *substitution* edit replaces one word with another word carrying the POS
    tag the grammar expected, so the parser needs to enumerate the vocabulary
    for a tag, which a flat ``NNS -> 'dog'`` style grammar makes awkward.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

EPSILON = "EPS"


@dataclass(frozen=True)
class Rule:
    """A single production ``lhs -> rhs``.

    ``rhs`` is a tuple of symbols; an empty tuple denotes an epsilon
    production.  Rules are frozen so they can be dict keys and so that a
    rule's identity is stable across the chart.
    """

    lhs: str
    rhs: tuple[str, ...]
    index: int = field(default=-1, compare=False)

    def __str__(self) -> str:  # pragma: no cover - display only
        rhs = " ".join(self.rhs) if self.rhs else "ε"
        return f"{self.lhs} -> {rhs}"

    def dotted(self, dot: int) -> str:
        """Render the rule with a dot at position ``dot`` (for chart dumps)."""
        parts = list(self.rhs)
        parts.insert(dot, "•")
        return f"{self.lhs} -> {' '.join(parts) if parts else '•'}"


class Grammar:
    """A context-free grammar indexed for Earley parsing."""

    def __init__(self, rules: list[Rule], start: str = "S") -> None:
        self.rules: list[Rule] = rules
        self.start: str = start

        # lhs -> rules, the index PREDICT needs.
        self._by_lhs: dict[str, list[Rule]] = defaultdict(list)
        for rule in rules:
            self._by_lhs[rule.lhs].append(rule)

        self.nonterminals: set[str] = {r.lhs for r in rules}
        # Any symbol appearing on a right-hand side but never on a left-hand
        # side is a pre-terminal (a POS tag supplied by the lexicon).
        rhs_symbols = {s for r in rules for s in r.rhs}
        self.preterminals: set[str] = rhs_symbols - self.nonterminals - {EPSILON}

        self._nullable: set[str] | None = None

    # -- lookup -------------------------------------------------------------

    def rules_for(self, lhs: str) -> list[Rule]:
        return self._by_lhs.get(lhs, [])

    def is_nonterminal(self, symbol: str) -> bool:
        return symbol in self.nonterminals

    def is_preterminal(self, symbol: str) -> bool:
        return symbol in self.preterminals

    # -- nullability --------------------------------------------------------

    def nullable(self) -> set[str]:
        """Non-terminals that can derive the empty string.

        Earley needs this to handle the well-known epsilon-production bug: a
        predicted nullable non-terminal must be completed immediately,
        otherwise items waiting on it never advance.
        """
        if self._nullable is not None:
            return self._nullable

        nullable: set[str] = set()
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if rule.lhs in nullable:
                    continue
                if all(sym in nullable for sym in rule.rhs):
                    nullable.add(rule.lhs)
                    changed = True
        self._nullable = nullable
        return nullable

    # -- construction -------------------------------------------------------

    @classmethod
    def from_string(cls, text: str, start: str = "S") -> "Grammar":
        rules: list[Rule] = []
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if "->" not in line:
                raise ValueError(f"malformed grammar line (no '->'): {raw_line!r}")
            lhs, rhs_text = line.split("->", 1)
            lhs = lhs.strip()
            if not lhs:
                raise ValueError(f"malformed grammar line (empty LHS): {raw_line!r}")
            for alternative in rhs_text.split("|"):
                symbols = tuple(alternative.split())
                if symbols == (EPSILON,):
                    symbols = ()
                rules.append(Rule(lhs, symbols, index=len(rules)))
        if not rules:
            raise ValueError("grammar contains no rules")
        return cls(rules, start=start)

    @classmethod
    def from_file(cls, path: str | Path, start: str = "S") -> "Grammar":
        return cls.from_string(Path(path).read_text(encoding="utf-8"), start=start)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Grammar(start={self.start!r}, rules={len(self.rules)}, "
            f"nonterminals={len(self.nonterminals)}, "
            f"preterminals={len(self.preterminals)})"
        )


@dataclass(frozen=True)
class LexEntry:
    """One word/tag pairing, with the lemma used for WordNet lookup."""

    word: str
    tag: str
    lemma: str


class Lexicon:
    """Word <-> POS-tag mapping with the reverse index error correction needs."""

    def __init__(self, entries: list[LexEntry]) -> None:
        self.entries = entries
        self._tags_for_word: dict[str, set[str]] = defaultdict(set)
        self._words_for_tag: dict[str, set[str]] = defaultdict(set)
        self._lemma: dict[tuple[str, str], str] = {}

        for entry in entries:
            self._tags_for_word[entry.word].add(entry.tag)
            self._words_for_tag[entry.tag].add(entry.word)
            self._lemma[(entry.word, entry.tag)] = entry.lemma

        self.vocabulary: set[str] = set(self._tags_for_word)

    # -- lookup -------------------------------------------------------------

    def tags(self, word: str) -> set[str]:
        return self._tags_for_word.get(word.lower(), set())

    def words(self, tag: str) -> set[str]:
        return self._words_for_tag.get(tag, set())

    def lemma(self, word: str, tag: str | None = None) -> str:
        word = word.lower()
        if tag is not None:
            return self._lemma.get((word, tag), word)
        for candidate_tag in self.tags(word):
            return self._lemma.get((word, candidate_tag), word)
        return word

    def knows(self, word: str) -> bool:
        return word.lower() in self.vocabulary

    # -- construction -------------------------------------------------------

    @classmethod
    def from_string(cls, text: str) -> "Lexicon":
        entries: list[LexEntry] = []
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = re.split(r"\t+|\s{2,}", line)
            fields = [f.strip() for f in fields if f.strip()]
            if len(fields) < 2:
                fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"malformed lexicon line: {raw_line!r}")
            word, tag = fields[0].lower(), fields[1]
            lemma = fields[2].lower() if len(fields) > 2 else word
            entries.append(LexEntry(word, tag, lemma))
        if not entries:
            raise ValueError("lexicon contains no entries")
        return cls(entries)

    @classmethod
    def from_file(cls, path: str | Path) -> "Lexicon":
        return cls.from_string(Path(path).read_text(encoding="utf-8"))

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Lexicon(words={len(self.vocabulary)}, tags={len(self._words_for_tag)}, "
            f"entries={len(self.entries)})"
        )


def check_coverage(grammar: Grammar, lexicon: Lexicon) -> list[str]:
    """Report pre-terminals the grammar uses but the lexicon cannot fill.

    An unfillable pre-terminal silently makes whole constructions unparsable,
    which is painful to debug through the chart, so the pipeline runs this at
    load time.
    """
    problems = []
    for tag in sorted(grammar.preterminals):
        if not lexicon.words(tag):
            problems.append(f"pre-terminal {tag!r} has no words in the lexicon")
    unused = sorted(set(lexicon._words_for_tag) - grammar.preterminals)
    for tag in unused:
        problems.append(f"lexicon tag {tag!r} is never used by the grammar")
    return problems
