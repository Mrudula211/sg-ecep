"""Download the WordNet corpus Lesk needs.

    python tools/setup_wordnet.py

The system runs without it -- `semantics.py` falls back to the bundled
ontology in `data/semantic_types.json`, and the test suite is written to pass
either way -- but WSD quality is much better with WordNet installed.
"""

import sys


def main() -> int:
    try:
        import nltk
    except ImportError:
        print("nltk is not installed.  Run:  pip install -r requirements.txt")
        return 1

    for package in ("wordnet", "omw-1.4"):
        print(f"downloading {package} ...")
        nltk.download(package, quiet=False)

    try:
        from nltk.corpus import wordnet as wn

        synset = wn.synsets("dog", "n")[0]
        print(f"\nWordNet is working: dog.n.01 = {synset.definition()}")
        return 0
    except Exception as error:
        print(f"\nWordNet still unavailable: {error}")
        print("The system will use the bundled ontology instead.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
