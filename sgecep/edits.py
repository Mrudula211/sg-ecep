"""Edit operations and the syntactic cost model.

The error-correcting parser repairs the *input* so that it fits the grammar.
Four operations suffice, and each carries a cost; the parser searches for the
cheapest repair, so the costs encode a linguistic prior about which mistakes
writers actually make.

``MATCH``
    The input token already carries the tag the grammar wanted.  Free.

``INFLECT``
    Replace the input token with a different inflection of the *same lemma*
    (``reads`` -> ``read``).  This is subject-verb agreement repair, by far
    the most common error in learner text, so it is deliberately much cheaper
    than a general substitution.  Separating it out is what stops the parser
    from "fixing" an agreement error by swapping the noun instead of the verb.

``SUBSTITUTE``
    Replace the input token with a word of a different lemma.

``DELETE``
    Drop a spurious input token (``the the dog``).

``INSERT``
    Supply a token the grammar requires and the input lacks (a missing
    determiner).  Insertion consumes no input, so its cost must be strictly
    positive or the parser would insert unboundedly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EditKind(str, Enum):
    MATCH = "match"
    INFLECT = "inflect"
    SUBSTITUTE = "substitute"
    DELETE = "delete"
    INSERT = "insert"


FUNCTION_TAGS: frozenset[str] = frozenset(
    {"DTS", "DTP", "IN", "MD", "CMP", "RB", "DEG", "PRPS", "PRPP", "PRPO"}
)
"""Closed-class tags.

Inserting or deleting a function word is a small, local repair.  Inserting a
*content* word invents information the writer never supplied, and deleting one
throws information away, so both are penalised heavily.  Without this split
the parser happily "repairs" a missing determiner by deleting the noun
instead, because a flat cost model makes those two repairs tie.
"""


@dataclass(frozen=True)
class CostModel:
    """Costs for the edit operations, split by open vs. closed word class.

    Defaults are set so that, in order of preference, the parser repairs by:
    re-inflecting (0.4) < adding or dropping a function word (0.6) <
    swapping one word for another (1.0) < inventing or discarding a content
    word (1.6).  That ordering is the linguistic prior; everything else the
    parser does follows from it.
    """

    match: float = 0.0
    inflect: float = 0.4
    substitute: float = 1.0
    delete_function: float = 0.6
    delete_content: float = 1.6
    insert_function: float = 0.6
    insert_content: float = 1.6
    delete_unknown: float = 0.6
    function_tags: frozenset[str] = FUNCTION_TAGS

    def insert_cost(self, tag: str) -> float:
        """Cost of supplying a token of ``tag`` that the input lacks."""
        return (
            self.insert_function if tag in self.function_tags else self.insert_content
        )

    def delete_cost(self, tags: set[str] | frozenset[str]) -> float:
        """Cost of dropping an input token carrying ``tags``.

        A token is treated as a function word if *any* of its readings is
        closed-class, which keeps genuinely ambiguous items such as ``that``
        (determiner or complementiser) cheap to remove.
        """
        if not tags:
            return self.delete_unknown
        if tags & self.function_tags:
            return self.delete_function
        return self.delete_content

    def validate(self) -> None:
        if min(self.insert_function, self.insert_content) <= 0:
            raise ValueError("insert costs must be > 0 or the parser will not terminate")
        if min(self.delete_function, self.delete_content, self.delete_unknown) <= 0:
            raise ValueError("delete costs must be > 0 or the parser will not terminate")
        if self.inflect > self.substitute:
            raise ValueError("inflect should not cost more than a general substitution")
        if self.insert_function > self.insert_content:
            raise ValueError("inserting a function word should be the cheaper insertion")


@dataclass(frozen=True)
class Edit:
    """One repair applied to the input, recorded for the edit script."""

    kind: EditKind
    position: int                 # index into the ORIGINAL token sequence
    original: str | None = None   # token that was there (None for INSERT)
    tag: str | None = None        # pre-terminal the grammar required
    replacement: str | None = None  # word finally chosen (filled by the reranker)

    def describe(self) -> str:
        if self.kind is EditKind.DELETE:
            return f"delete {self.original!r} at position {self.position}"
        if self.kind is EditKind.INSERT:
            target = self.replacement or f"<{self.tag}>"
            return f"insert {target!r} before position {self.position}"
        if self.kind is EditKind.INFLECT:
            target = self.replacement or f"<{self.tag}>"
            return (
                f"re-inflect {self.original!r} -> {target!r} "
                f"at position {self.position}"
            )
        if self.kind is EditKind.SUBSTITUTE:
            target = self.replacement or f"<{self.tag}>"
            return f"replace {self.original!r} -> {target!r} at position {self.position}"
        return f"match {self.original!r} at position {self.position}"

    def with_replacement(self, word: str) -> "Edit":
        return Edit(self.kind, self.position, self.original, self.tag, word)


def summarise(edits: list[Edit]) -> str:
    """One-line summary of an edit script, e.g. ``1 inflect, 1 insert``."""
    real = [e for e in edits if e.kind is not EditKind.MATCH]
    if not real:
        return "no edits (already grammatical)"
    counts: dict[str, int] = {}
    for edit in real:
        counts[edit.kind.value] = counts.get(edit.kind.value, 0) + 1
    return ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
