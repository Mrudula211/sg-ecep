"""Semantic plausibility: shallow SRL plus selectional-restriction checking.

Pipeline for one candidate parse:

1. **Shallow semantic role labelling** -- read predicate-argument structure
   straight off the parse tree.  Because the tree comes from a hand-written
   CFG whose verb-phrase frames are known, the subject (ARG0), direct object
   (ARG1) and recipient (ARG2) can be located by structural rules rather than
   a trained classifier.
2. **Word sense disambiguation** -- run Lesk on each argument head noun,
   using the rest of the sentence as context.
3. **Selectional-restriction checking** -- test whether the chosen sense
   belongs to a semantic type the verb permits in that role.

Step 2 matters more than it looks.  A checker that tests every sense of a noun
and accepts if *any* passes would let ``the cat ate the homework`` through,
because some sense of almost any noun can be coerced into almost any type.
Committing to one sense first is what gives the checker teeth -- and it is
also why a bad disambiguation must only ever weaken the score, never veto the
parse outright.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .grammar import Lexicon
from .lesk import LeskDisambiguator
from .trees import Tree

NOUN_TAGS = frozenset({"NNS", "NNP", "NNPS"})
PRONOUN_TAGS = frozenset({"PRPS", "PRPP", "PRPO"})
SUBCAT_CLASSES = frozenset("ITDSC")       # see tools/build_lexicon.py
INFLECTIONS = frozenset("ZPDB")


def is_verb_tag(tag: str | None) -> bool:
    """True for a generated verb tag such as ``VTZ`` or ``VID``."""
    return (
        tag is not None
        and len(tag) == 3
        and tag[0] == "V"
        and tag[1] in SUBCAT_CLASSES
        and tag[2] in INFLECTIONS
    )

# Pronouns carry their semantic type lexically rather than through WordNet.
PRONOUN_TYPES: dict[str, tuple[str, ...]] = {
    "he": ("person",),
    "she": ("person",),
    "i": ("person",),
    "we": ("person",),
    "you": ("person",),
    "him": ("person",),
    "her": ("person",),
    "us": ("person",),
    "me": ("person",),
    "they": ("person", "physical_entity"),
    "them": ("person", "physical_entity"),
    "it": ("physical_entity",),
}


@dataclass
class Argument:
    """One predicate argument found in the tree."""

    role: str                 # ARG0 | ARG1 | ARG2
    head: str | None          # head word, or None if the slot is unfilled
    tag: str | None
    is_slot: bool = False


@dataclass
class Proposition:
    """A verb and its arguments, as read off one parse tree."""

    predicate: str | None
    predicate_tag: str | None
    arguments: list[Argument] = field(default_factory=list)
    instrument: str | None = None   # head of a `with`-PP, if present
    pp_attachment: str | None = None  # "verb" or "noun"

    def argument(self, role: str) -> Argument | None:
        for arg in self.arguments:
            if arg.role == role:
                return arg
        return None

    def __str__(self) -> str:  # pragma: no cover - display only
        parts = [f"{a.role}={a.head or '□'}" for a in self.arguments]
        if self.instrument:
            parts.append(f"INSTRUMENT={self.instrument}")
        return f"{self.predicate or '□'}({', '.join(parts)})"


@dataclass
class SemanticReport:
    """The semantic verdict on one candidate parse."""

    score: float                      # 0.0 (implausible) .. 1.0 (fully plausible)
    propositions: list[Proposition]
    checks: int = 0                   # how many restrictions could be tested
    passed: float = 0.0               # how many of them held
    satisfied: list[str] = field(default_factory=list)
    violated: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
    senses: dict[str, str] = field(default_factory=dict)

    def explain(self) -> str:
        lines = [
            f"semantic score {self.score:.2f} "
            f"({self.passed:.0f}/{self.checks} restrictions satisfied)"
        ]
        for prop in self.propositions:
            lines.append(f"  predicate-argument: {prop}")
        for item in self.satisfied:
            lines.append(f"  satisfied : {item}")
        for item in self.violated:
            lines.append(f"  VIOLATED  : {item}")
        for item in self.unchecked:
            lines.append(f"  unchecked : {item}")
        for word, sense in self.senses.items():
            lines.append(f"  sense     : {word} -> {sense}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Semantic type assignment
# ---------------------------------------------------------------------------


class SemanticTypeIndex:
    """Maps a noun (or a noun sense) to coarse semantic types.

    Two backends, deliberately interchangeable:

    ``wordnet``
        Disambiguate with Lesk, then test the chosen sense's hypernym closure
        against the anchor synsets of each coarse type.

    ``fallback``
        A bundled lemma -> types table.  Keeps the system runnable without a
        WordNet download and keeps the unit tests independent of a 10 MB
        corpus.
    """

    def __init__(
        self,
        anchors: dict[str, list[str]],
        fallback: dict[str, list[str]],
        disambiguator: LeskDisambiguator | None = None,
    ) -> None:
        self.anchors = anchors
        self.fallback = {k: tuple(v) for k, v in fallback.items()}
        self.lesk = disambiguator or LeskDisambiguator()

    @classmethod
    def from_file(
        cls, path: str | Path, disambiguator: LeskDisambiguator | None = None
    ) -> "SemanticTypeIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["anchors"], data["fallback"], disambiguator)

    @property
    def uses_wordnet(self) -> bool:
        return self.lesk.available

    def types_of_sense(self, synset_name: str) -> frozenset[str]:
        closure = self.lesk.hypernym_closure(synset_name)
        if not closure:
            return frozenset()
        found = {
            coarse
            for coarse, anchor_names in self.anchors.items()
            if any(anchor in closure for anchor in anchor_names)
        }
        return frozenset(found)

    def types_of_word(
        self,
        word: str,
        context: list[str],
        lemma: str | None = None,
        tag: str | None = None,
    ) -> tuple[frozenset[str], str | None]:
        """Coarse types of ``word`` in ``context``, plus the sense chosen."""
        key = (lemma or word).lower()
        if word.lower() in PRONOUN_TYPES:
            return frozenset(PRONOUN_TYPES[word.lower()]), None

        # Proper nouns bypass WordNet entirely.  WordNet has common-noun
        # entries for many given names -- john.n.01 is a lavatory, mark.n.01 a
        # blemish -- and Lesk has no way to tell that the token is a name, so
        # it confidently returns the wrong type.
        if tag == "NNPS":
            for candidate in (key, word.lower()):
                if candidate in self.fallback:
                    return frozenset(self.fallback[candidate]), None
            return frozenset({"person"}), None

        if self.uses_wordnet:
            sense = self.lesk.disambiguate(key, context, pos="n")
            if sense is not None:
                types = self.types_of_sense(sense.synset_name)
                if types:
                    return types, sense.synset_name
                # WordNet knows the word but none of our anchors dominate it;
                # fall through to the bundled table rather than report nothing.

        for candidate in (key, word.lower()):
            if candidate in self.fallback:
                return frozenset(self.fallback[candidate]), None
        return frozenset(), None


# ---------------------------------------------------------------------------
# Shallow SRL
# ---------------------------------------------------------------------------


def _head_noun(node: Tree) -> Tree | None:
    """Rightmost noun or pronoun leaf inside an NP, skipping any PP modifier.

    Taking the rightmost noun *before* a PP gets ``the man with the telescope``
    right: the head is ``man``, not ``telescope``.
    """
    candidates = [
        leaf
        for leaf in node.leaves()
        if leaf.label in NOUN_TAGS or leaf.label in PRONOUN_TAGS
    ]
    if not candidates:
        return None

    nominal = [child for child in node.children if child.label in ("NOMS", "NOMP")]
    if not nominal:
        for child in node.children:
            nominal.extend(
                sub for sub in child.children if sub.label in ("NOMS", "NOMP")
            )
    if nominal:
        inner = [
            leaf
            for child in nominal
            for leaf in child.leaves()
            if leaf.label in NOUN_TAGS
        ]
        pp_leaves = {
            id(leaf)
            for child in nominal
            for sub in child.children
            if sub.label == "PP"
            for leaf in sub.leaves()
        }
        inner = [leaf for leaf in inner if id(leaf) not in pp_leaves]
        if inner:
            return inner[0]
    return candidates[0]


def _np_children(node: Tree) -> list[Tree]:
    return [child for child in node.children if child.label in ("NPS", "NPP", "NPO")]


def _unwrap(node: Tree) -> Tree:
    """Collapse the NPO indirection introduced for object position."""
    while node.label in ("NPO", "NPS", "NPP") and len(node.children) == 1:
        if node.children[0].label not in ("NPS", "NPP", "NPO"):
            break
        node = node.children[0]
    return node


def extract_propositions(tree: Tree) -> list[Proposition]:
    """Read predicate-argument structure off a parse tree."""
    propositions: list[Proposition] = []

    def visit(node: Tree) -> None:
        if node.label == "S":
            subject = None
            vp = None
            for child in node.children:
                if child.label in ("NPS", "NPP") and subject is None:
                    subject = child
                elif child.label in ("VPS", "VPP", "VPIMP"):
                    vp = child
            if vp is not None:
                propositions.append(_proposition(vp, subject))
        for child in node.children:
            visit(child)

    visit(tree)
    return propositions


def _proposition(vp: Tree, subject: Tree | None) -> Proposition:
    # Adjunction wraps the core verb phrase in further VP nodes, so the chain
    # has to be collected rather than just descended: the verb and its objects
    # live on the innermost node, while adjoined PPs and adverbs hang off the
    # outer ones.
    chain = [vp]
    inner = vp
    while True:
        nested = [c for c in inner.children if c.label in ("VPS", "VPP", "VPIMP")]
        if not nested:
            break
        inner = nested[0]
        chain.append(inner)

    # With a modal ("should read"), the content verb is the base form after
    # it, not the modal, so the modal is simply skipped: MD is not a verb tag.
    verbs = [leaf for leaf in inner.leaves() if is_verb_tag(leaf.label)]
    verb_leaf = verbs[0] if verbs else None

    prop = Proposition(
        predicate=verb_leaf.word if verb_leaf else None,
        predicate_tag=verb_leaf.label if verb_leaf else None,
    )

    if subject is not None:
        head = _head_noun(subject)
        prop.arguments.append(
            Argument(
                "ARG0",
                head.word if head else None,
                head.label if head else None,
                is_slot=bool(head and head.word is None),
            )
        )

    objects = [_unwrap(child) for child in _np_children(inner)]
    if len(objects) == 1:
        head = _head_noun(objects[0])
        prop.arguments.append(
            Argument(
                "ARG1",
                head.word if head else None,
                head.label if head else None,
                is_slot=bool(head and head.word is None),
            )
        )
    elif len(objects) >= 2:
        # Ditransitive: "gave [the students] [a book]" -> recipient then theme.
        recipient, theme = objects[0], objects[1]
        for role, np in (("ARG2", recipient), ("ARG1", theme)):
            head = _head_noun(np)
            prop.arguments.append(
                Argument(
                    role,
                    head.word if head else None,
                    head.label if head else None,
                    is_slot=bool(head and head.word is None),
                )
            )

    # A `with`-PP directly under the VP is an instrument candidate; one inside
    # the object NP is a noun modifier.  Which one the tree chose is exactly
    # the PP-attachment ambiguity the semantic layer is asked to resolve.
    for node in chain:
        for child in node.children:
            if child.label == "PP":
                prep = next(
                    (leaf for leaf in child.leaves() if leaf.label == "IN"), None
                )
                if prep and prep.word == "with":
                    head = _head_noun(child)
                    prop.instrument = head.word if head else None
                    prop.pp_attachment = "verb"
    if prop.pp_attachment is None:
        for np in objects:
            for child in np.children:
                for sub in child.children if child.label in ("NOMS", "NOMP") else []:
                    if sub.label == "PP":
                        prep = next(
                            (leaf for leaf in sub.leaves() if leaf.label == "IN"), None
                        )
                        if prep and prep.word == "with":
                            prop.pp_attachment = "noun"
    return prop


# ---------------------------------------------------------------------------
# Selectional restriction checking
# ---------------------------------------------------------------------------


class SelectionalChecker:
    """Scores a parse tree for semantic plausibility."""

    def __init__(
        self,
        restrictions: dict,
        types: SemanticTypeIndex,
        lexicon: Lexicon | None = None,
    ) -> None:
        self.verbs: dict[str, dict] = restrictions.get("verbs", {})
        self.instrument_prefs: dict = {
            k: v
            for k, v in restrictions.get("instrument_preferences", {}).items()
            if not k.startswith("_")
        }
        self.types = types
        self.lexicon = lexicon

    @classmethod
    def from_files(
        cls,
        restrictions_path: str | Path,
        types_path: str | Path,
        lexicon: Lexicon | None = None,
        disambiguator: LeskDisambiguator | None = None,
    ) -> "SelectionalChecker":
        restrictions = json.loads(Path(restrictions_path).read_text(encoding="utf-8"))
        types = SemanticTypeIndex.from_file(types_path, disambiguator)
        return cls(restrictions, types, lexicon)

    def _lemma(self, word: str, tag: str | None) -> str:
        if self.lexicon is not None:
            return self.lexicon.lemma(word, tag)
        return word.lower()

    def score(self, tree: Tree) -> SemanticReport:
        propositions = extract_propositions(tree)
        context = [w for w in tree.yield_words() if not w.startswith("<")]

        satisfied: list[str] = []
        violated: list[str] = []
        unchecked: list[str] = []
        senses: dict[str, str] = {}
        checks = 0
        passed = 0.0

        for prop in propositions:
            if prop.predicate is None:
                unchecked.append("predicate is an unfilled slot")
                continue
            verb_lemma = self._lemma(prop.predicate, prop.predicate_tag)
            frame = self.verbs.get(verb_lemma)
            if frame is None:
                unchecked.append(f"no restrictions recorded for verb {verb_lemma!r}")
                continue

            for arg in prop.arguments:
                allowed = frame.get(arg.role)
                if allowed is None:
                    continue
                if arg.head is None:
                    unchecked.append(f"{verb_lemma}.{arg.role} is an unfilled slot")
                    continue

                arg_lemma = self._lemma(arg.head, arg.tag)
                arg_types, sense = self.types.types_of_word(
                    arg.head, context, lemma=arg_lemma, tag=arg.tag
                )
                if sense:
                    senses[arg.head] = sense
                if not arg_types:
                    unchecked.append(f"no semantic type known for {arg.head!r}")
                    continue

                checks += 1
                if arg_types & set(allowed):
                    passed += 1.0
                    satisfied.append(
                        f"{verb_lemma}.{arg.role} accepts {arg.head!r} "
                        f"({'/'.join(sorted(arg_types & set(allowed)))})"
                    )
                else:
                    violated.append(
                        f"{verb_lemma}.{arg.role} requires {'/'.join(allowed)} "
                        f"but {arg.head!r} is {'/'.join(sorted(arg_types))}"
                    )

            # Instrument plausibility feeds PP-attachment disambiguation.
            if prop.instrument and prop.pp_attachment == "verb":
                prefs = self.instrument_prefs.get(verb_lemma)
                inst_types, sense = self.types.types_of_word(
                    prop.instrument,
                    context,
                    lemma=self._lemma(prop.instrument, None),
                )
                if sense:
                    senses[prop.instrument] = sense
                if prefs and inst_types:
                    checks += 1
                    if inst_types & set(prefs):
                        passed += 1.0
                        satisfied.append(
                            f"{verb_lemma} can take {prop.instrument!r} as instrument"
                        )
                    else:
                        violated.append(
                            f"{prop.instrument!r} is an implausible instrument "
                            f"for {verb_lemma}"
                        )

        # Neutral (0.5) when nothing could be checked: absence of evidence must
        # not look like evidence of plausibility, nor of implausibility.
        score = 0.5 if checks == 0 else passed / checks
        return SemanticReport(
            score=score,
            propositions=propositions,
            checks=checks,
            passed=passed,
            satisfied=satisfied,
            violated=violated,
            unchecked=unchecked,
            senses=senses,
        )
