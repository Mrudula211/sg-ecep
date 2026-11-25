"""Parse-tree representation and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Tree:
    """A parse tree node.

    A node is either internal (``children`` non-empty, ``word`` None) or a
    leaf (``word`` set).  ``origin`` records how a leaf came to be in the tree
    so the demo can colour repaired words differently from original ones:

    ``"input"``    the word was present in the input and matched
    ``"inserted"`` the parser inserted this word (nothing consumed)
    ``"substituted"`` this word replaced a different input word
    """

    label: str
    children: list["Tree"] = field(default_factory=list)
    word: str | None = None
    origin: str = "input"
    terminal: bool = False

    @property
    def is_leaf(self) -> bool:
        # A leaf is a terminal node whether or not a word has been chosen yet:
        # an unfilled *slot* left by the error-correcting parser is still a
        # leaf, and must survive traversal so the reranker can fill it.
        return self.terminal or (self.word is not None and not self.children)

    @property
    def is_slot(self) -> bool:
        return self.is_leaf and self.word is None

    def leaves(self) -> list["Tree"]:
        if self.is_leaf:
            return [self]
        out: list[Tree] = []
        for child in self.children:
            out.extend(child.leaves())
        return out

    def yield_words(self) -> list[str]:
        """Surface words; slots render as ``<TAG>`` so nothing silently vanishes."""
        return [
            leaf.word if leaf.word is not None else f"<{leaf.label}>"
            for leaf in self.leaves()
        ]

    def filled(self) -> bool:
        return all(leaf.word is not None for leaf in self.leaves())

    def copy(self) -> "Tree":
        return Tree(
            label=self.label,
            children=[c.copy() for c in self.children],
            word=self.word,
            origin=self.origin,
            terminal=self.terminal,
        )

    # -- rendering ----------------------------------------------------------

    def to_bracketed(self) -> str:
        """Penn-Treebank style bracketing, e.g. ``(S (NPS (DTS the) ...))``."""
        if self.is_leaf:
            return f"({self.label} {self.word if self.word is not None else '□'})"
        inner = " ".join(child.to_bracketed() for child in self.children)
        return f"({self.label} {inner})"

    def pretty(self, indent: str = "", is_last: bool = True) -> str:
        """ASCII tree drawing, one node per line."""
        connector = "" if not indent and is_last else ("└── " if is_last else "├── ")
        if self.is_leaf:
            mark = {"inserted": " [+]", "substituted": " [~]"}.get(self.origin, "")
            shown = self.word if self.word is not None else f"<{self.label} slot>"
            line = f"{indent}{connector}{self.label} : {shown}{mark}"
            return line
        lines = [f"{indent}{connector}{self.label}"]
        child_indent = indent + ("" if not connector else ("    " if is_last else "│   "))
        for i, child in enumerate(self.children):
            lines.append(child.pretty(child_indent, i == len(self.children) - 1))
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.to_bracketed()


def find_nodes(tree: Tree, label: str) -> list[Tree]:
    """Every node in ``tree`` carrying ``label`` (pre-order)."""
    found: list[Tree] = []
    if tree.label == label:
        found.append(tree)
    for child in tree.children:
        found.extend(find_nodes(child, label))
    return found


def tagged_words(tree: Tree) -> list[tuple[str, str]]:
    """``(word, pos_tag)`` pairs in surface order."""
    return [(leaf.word, leaf.label) for leaf in tree.leaves() if leaf.word is not None]


def slots(tree: Tree) -> list[Tree]:
    """Leaf nodes still awaiting a word."""
    return [leaf for leaf in tree.leaves() if leaf.is_slot]
