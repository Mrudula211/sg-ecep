"""Generate ``data/lexicon.tsv``.

The verb inventory is the part of the lexicon that must stay consistent with
the grammar: every verb carries both an inflection (3sg present, non-3sg
present, past, base) and a subcategorisation class (intransitive, transitive,
ditransitive, sentential-complement, copular), and the grammar's verb-phrase
frames are keyed on the resulting tag.  Writing those 20 verb tags out by hand
invites exactly the kind of silent mismatch that makes a whole construction
unparsable, so they are generated instead.

Run with::

    python tools/build_lexicon.py
"""

from __future__ import annotations

from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# -- verbs ------------------------------------------------------------------
# lemma -> (3sg present, non-3sg present, past, base)
INFLECTIONS: dict[str, tuple[str, str, str, str]] = {
    "sleep": ("sleeps", "sleep", "slept", "sleep"),
    "walk": ("walks", "walk", "walked", "walk"),
    "run": ("runs", "run", "ran", "run"),
    "eat": ("eats", "eat", "ate", "eat"),
    "read": ("reads", "read", "read", "read"),
    "write": ("writes", "write", "wrote", "write"),
    "see": ("sees", "see", "saw", "see"),
    "watch": ("watches", "watch", "watched", "watch"),
    "drink": ("drinks", "drink", "drank", "drink"),
    "cut": ("cuts", "cut", "cut", "cut"),
    "open": ("opens", "open", "opened", "open"),
    "drive": ("drives", "drive", "drove", "drive"),
    "break": ("breaks", "break", "broke", "break"),
    "teach": ("teaches", "teach", "taught", "teach"),
    "study": ("studies", "study", "studied", "study"),
    "like": ("likes", "like", "liked", "like"),
    "know": ("knows", "know", "knew", "know"),
    "think": ("thinks", "think", "thought", "think"),
    "give": ("gives", "give", "gave", "give"),
    "send": ("sends", "send", "sent", "send"),
    "tell": ("tells", "tell", "told", "tell"),
    "have": ("has", "have", "had", "have"),
    "be": ("is", "are", "was", "be"),
}

# subcategorisation class -> tag prefix
CLASS_PREFIX = {
    "intransitive": "VI",
    "transitive": "VT",
    "ditransitive": "VD",
    "sentential": "VS",
    "copular": "VC",
}

# lemma -> the classes it belongs to (a verb may belong to several)
SUBCAT: dict[str, tuple[str, ...]] = {
    "sleep": ("intransitive",),
    "walk": ("intransitive",),
    "run": ("intransitive",),
    "eat": ("intransitive", "transitive"),
    "read": ("intransitive", "transitive"),
    "write": ("intransitive", "transitive"),
    "see": ("transitive", "sentential"),
    "watch": ("transitive",),
    "drink": ("intransitive", "transitive"),
    "cut": ("transitive",),
    "open": ("transitive",),
    "drive": ("intransitive", "transitive"),
    "break": ("transitive",),
    "teach": ("transitive", "ditransitive"),
    "study": ("intransitive", "transitive"),
    "like": ("transitive",),
    "know": ("transitive", "sentential"),
    "think": ("sentential",),
    "give": ("ditransitive",),
    "send": ("transitive", "ditransitive"),
    "tell": ("ditransitive", "sentential"),
    "have": ("transitive",),
    "be": ("copular",),
}

# Suffix order matches INFLECTIONS: Z = 3sg present, P = non-3sg present,
# D = past, B = base form.
SUFFIXES = ("Z", "P", "D", "B")

# -- closed classes and open-class nouns -------------------------------------
DETERMINERS_SG = ["a", "an", "the", "this", "that", "every", "each"]
DETERMINERS_PL = ["the", "these", "those", "some", "many", "two", "three", "several", "all"]
PRONOUN_SUBJ_SG = ["he", "she", "it", "i"]
PRONOUN_SUBJ_PL = ["they", "we", "you"]
PRONOUN_OBJ = ["him", "her", "it", "them", "us", "me", "you"]
PROPER = ["john", "mary", "atharva", "mumbai", "pune"]
MODALS = ["can", "will", "should", "must", "might", "could"]
PREPOSITIONS = ["in", "on", "at", "with", "to", "from", "under", "near", "for"]
COMPLEMENTISERS = ["that"]
DEGREE = ["very", "quite", "rather"]
ADVERBS = ["quickly", "slowly", "quietly", "often", "always", "never", "yesterday", "today"]
ADJECTIVES = [
    "big", "small", "old", "young", "red", "green", "hungry", "happy", "sad",
    "tired", "clever", "heavy", "sharp", "fresh", "tall",
]

# singular noun -> plural form (None means the noun has no plural here)
NOUNS: dict[str, str | None] = {
    "student": "students", "teacher": "teachers", "child": "children",
    "boy": "boys", "girl": "girls", "man": "men", "woman": "women",
    "engineer": "engineers", "doctor": "doctors", "nurse": "nurses",
    "patient": "patients",
    "dog": "dogs", "cat": "cats", "bird": "birds", "horse": "horses",
    "book": "books", "letter": "letters", "report": "reports",
    "paper": "papers", "homework": None, "story": "stories",
    "question": "questions", "answer": "answers", "idea": "ideas",
    "apple": "apples", "pizza": "pizzas", "sandwich": "sandwiches",
    "bread": None, "cake": "cakes", "soup": "soups", "water": None, "milk": None,
    "telescope": "telescopes", "camera": "cameras", "computer": "computers",
    "pen": "pens", "knife": "knives", "hammer": "hammers", "spoon": "spoons",
    "car": "cars", "bus": "buses", "train": "trains",
    "table": "tables", "chair": "chairs", "door": "doors",
    "window": "windows", "box": "boxes",
    "school": "schools", "library": "libraries", "park": "parks",
    "kitchen": "kitchens", "garden": "gardens", "room": "rooms",
    "river": "rivers", "bank": "banks", "hill": "hills",
    "rock": "rocks", "stone": "stones", "tree": "trees",
}

HEADER = """\
# Lexicon: word <TAB> POS-tag <TAB> lemma
#
# GENERATED FILE -- edit tools/build_lexicon.py and re-run it instead.
#
# Tag inventory
#   DTS/DTP    determiner, singular / plural
#   NNS/NNP    common noun, singular / plural
#   NNPS       proper noun
#   PRPS/PRPP  nominative pronoun, singular / plural   (subject position)
#   PRPO       accusative pronoun                      (object position)
#   JJ         adjective        DEG  degree modifier ("very")
#   RB         adverb           IN   preposition      CMP  complementiser
#   MD         modal
#   V<class><inflection> where class is
#       I intransitive, T transitive, D ditransitive,
#       S sentential-complement, C copular
#   and inflection is
#       Z 3sg present, P non-3sg present, D past, B base form
#   e.g. VTZ = transitive, 3rd person singular present ("eats")
"""


def build() -> str:
    rows: list[tuple[str, str, str]] = []

    def add(word: str, tag: str, lemma: str | None = None) -> None:
        rows.append((word.lower(), tag, (lemma or word).lower()))

    sections: list[str] = [HEADER]

    def section(title: str) -> None:
        nonlocal rows
        if rows:
            sections.append(
                "\n".join(f"{w}\t{t}\t{l}" for w, t, l in rows)
            )
            rows = []
        sections.append(f"\n# -- {title} " + "-" * max(0, 60 - len(title)))

    section("determiners")
    for word in DETERMINERS_SG:
        add(word, "DTS")
    for word in DETERMINERS_PL:
        add(word, "DTP")

    section("pronouns")
    for word in PRONOUN_SUBJ_SG:
        add(word, "PRPS")
    for word in PRONOUN_SUBJ_PL:
        add(word, "PRPP")
    for word in PRONOUN_OBJ:
        add(word, "PRPO")

    section("nouns")
    for singular, plural in NOUNS.items():
        add(singular, "NNS", singular)
        if plural:
            add(plural, "NNP", singular)
    for word in PROPER:
        add(word, "NNPS")

    section("verbs (generated from the subcategorisation table)")
    for lemma, classes in SUBCAT.items():
        forms = INFLECTIONS[lemma]
        for cls in classes:
            prefix = CLASS_PREFIX[cls]
            for form, suffix in zip(forms, SUFFIXES):
                add(form, f"{prefix}{suffix}", lemma)
    # "were" is the plural past of be and shares the VCD slot with "was".
    add("were", "VCD", "be")

    section("modals, adjectives, adverbs, prepositions")
    for word in MODALS:
        add(word, "MD")
    for word in ADJECTIVES:
        add(word, "JJ")
    for word in DEGREE:
        add(word, "DEG")
    for word in ADVERBS:
        add(word, "RB")
    for word in PREPOSITIONS:
        add(word, "IN")
    for word in COMPLEMENTISERS:
        add(word, "CMP")

    section("end")
    sections = [s for s in sections if not s.endswith("end " + "-" * 57)]
    return "\n".join(sections).rstrip() + "\n"


if __name__ == "__main__":
    target = DATA / "lexicon.tsv"
    text = build()
    target.write_text(text, encoding="utf-8")
    entries = sum(1 for line in text.splitlines() if line and not line.startswith("#"))
    print(f"wrote {target} ({entries} entries)")
