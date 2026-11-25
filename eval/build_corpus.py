"""Build ``data/corpus.txt``, the language-model training corpus.

Sentences are sampled from the CFG and then filtered through the
selectional-restriction checker, so the corpus contains only strings that are
both grammatical *and* semantically plausible.  Training the LM on grammar
output rather than external text matters: an LM trained on general English
would assign low probability to perfectly good sentences here purely because
this grammar's vocabulary is small, and the LM's whole job in the reranker is
to compare candidates *within* that vocabulary.

Run with::

    python -m eval.build_corpus --count 900
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sgecep.earley import EarleyParser
from sgecep.generate import Generator
from sgecep.grammar import Grammar, Lexicon
from sgecep.lesk import LeskDisambiguator
from sgecep.semantics import SelectionalChecker

DATA = Path(__file__).resolve().parent.parent / "data"

HEADER = """\
# Language-model training corpus -- GENERATED FILE.
#
# Produced by sampling derivations from data/grammar.cfg and keeping only
# those whose predicate-argument structure satisfies every applicable
# selectional restriction in data/selectional_restrictions.json.
#
# Regenerate with:  python -m eval.build_corpus
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=900, help="sentences to keep")
    parser.add_argument("--samples", type=int, default=20000, help="sampling budget")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=10)
    parser.add_argument(
        "--no-wordnet",
        action="store_true",
        help="use the bundled ontology instead of WordNet",
    )
    args = parser.parse_args()

    grammar = Grammar.from_file(DATA / "grammar.cfg")
    lexicon = Lexicon.from_file(DATA / "lexicon.tsv")
    generator = Generator(grammar, lexicon, seed=args.seed)
    earley = EarleyParser(grammar, lexicon)

    disambiguator = LeskDisambiguator()
    if args.no_wordnet:
        disambiguator = _Offline()
    checker = SelectionalChecker.from_files(
        DATA / "selectional_restrictions.json",
        DATA / "semantic_types.json",
        lexicon=lexicon,
        disambiguator=disambiguator,
    )

    samples = generator.sample_many(
        args.samples, min_length=args.min_length, max_length=args.max_length
    )
    print(f"sampled {len(samples)} unique sentences from the grammar")

    kept: set[str] = set()
    rejected = 0
    for tokens in samples:
        trees = earley.parse(tokens, limit=1)
        if not trees:
            continue
        if checker.score(trees[0]).score >= 0.999:
            kept.add(" ".join(tokens))
        else:
            rejected += 1
        if len(kept) >= args.count:
            break

    target = DATA / "corpus.txt"
    target.write_text(HEADER + "\n".join(sorted(kept)) + "\n", encoding="utf-8")
    print(f"kept {len(kept)}, rejected {rejected} as semantically implausible")
    print(f"wrote {target}")


class _Offline(LeskDisambiguator):
    @property
    def available(self) -> bool:  # type: ignore[override]
        return False


if __name__ == "__main__":
    main()
