# SG-ECEP — Semantically-Guided Error-Correcting Earley Parser

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-78%20passing-brightgreen)
![NLTK](https://img.shields.io/badge/NLTK-WordNet-orange)

A parser that **repairs** ungrammatical sentences instead of rejecting them,
and uses **semantic plausibility** to choose between repairs that edit
distance alone cannot separate.

```
$ python -m sgecep.cli "the students reads the book"
input      : the students reads the book
corrected  : the student reads the book
edits      : 1 inflect
             - re-inflect 'students' -> 'student' at position 1
scores     : cost 0.40 | semantic 1.00 (2 checks) | lm -4.790
```

---

## The idea in one paragraph

A conventional Earley parser answers yes or no. This one answers *"no, but
here is the nearest sentence that would have been yes, and here is what I
changed."* It finds the cheapest sequence of edits — insert, delete,
substitute, re-inflect — that brings the input inside the grammar. When two
repairs cost exactly the same, a semantic layer (Lesk WSD plus selectional
restrictions over WordNet) and a Kneser-Ney language model decide between
them.

The design choice that makes the semantics load-bearing: the parser does not
emit repaired *strings*. It emits repair **templates** with unfilled slots,
recording only the part of speech the grammar demanded. Filling a noun slot is
a choice among sixty-odd nouns that are **identical in edit cost**, so edit
distance cannot decide it at all. The semantics is not re-ranking finished
output; it is choosing content.

## Quick start

```bash
git clone https://github.com/Mrudula211/sg-ecep.git
cd sg-ecep
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/setup_wordnet.py      # optional but recommended

python -m sgecep.cli "student reads the book"
python -m sgecep.cli --demo
python -m sgecep.cli --interactive
python -m sgecep.cli -v "she saw the man with the pizza"
python -m sgecep.cli --chart "the students reads the book"

pytest                             # 78 tests, ~8 seconds
python -m eval.run_eval --limit 150
```

The system runs **without WordNet** — it falls back to the bundled ontology in
`data/semantic_types.json`, and the test suite is written to pass either way.
WSD quality is noticeably better with WordNet installed.

## Demo

A two-minute narrated walkthrough is in
[`docs/media/sgecep-demo.mp4`](docs/media/sgecep-demo.mp4). The sentences used
by `python -m sgecep.cli --demo` live in
[`demo/demo_sentences.txt`](demo/demo_sentences.txt).

## Results

Grammatical sentences are corrupted on purpose; the system is scored on how
often it recovers the original. Metrics are computed over the genuinely
ungrammatical cases only — roughly 30% of corruptions yield *another*
grammatical sentence, and the system is right to leave those alone.

| configuration | exact | oracle | semantically valid |
|---|---|---|---|
| baseline (plain Earley) | 0.0% | 0.0% | 0.0% |
| syntax-only | 41.2% | 51.2% | 87.5% |
| syntax + LM | 47.1% | 59.4% | 91.2% |
| **full (+ semantics)** | **47.2%** | **59.7%** | **93.5%** |

695 ungrammatical cases, derived from 200 gold sentences (1000 corruptions,
305 of which left the sentence grammatical).

The semantic layer lifts validity by 6.0 points while moving exact recovery
by 0.1 — and that is the honest finding, not a disappointment. Semantics does
not help you guess *which* word the writer originally used; it makes the
correction **plausible**. When a slot admits sixty nouns, insisting on the
gold word penalises corrections that are perfectly sensible but different.

Supporting numbers: Kneser-Ney reaches perplexity 52.7 on a held-out split
against 61.1 for add-0.1 and 95.7 for add-1.0; and the parser returned more than one
minimum-cost repair in 304 of 695 cases (44%), so the tie-breaking problem
this project addresses is frequent rather than contrived.

The full evaluation log, including per-corruption-type breakdowns, is in
[`docs/EVALUATION_RESULTS.txt`](docs/EVALUATION_RESULTS.txt).

## How it works

```
input tokens
     │
     ├─ baseline Earley ──── grammatical? ──► yes ──► return unchanged
     │                            │
     │                            no
     │                            ▼
     ├─ error-correcting Earley chart  (insert / delete / substitute / inflect)
     │        │
     │        ▼
     │   repair templates with unfilled slots, cheapest first
     │        │
     ├─ slot filling ──────── language model picks words for open slots
     │        │               (INFLECT slots are locked to the original lemma)
     │        ▼
     ├─ shallow SRL ───────── predicate(ARG0, ARG1, ARG2) off the parse tree
     ├─ Lesk WSD ──────────── one WordNet sense per argument head
     ├─ selectional check ─── sense's hypernym closure ∩ verb's permitted types
     │        │
     └─ rerank ────────────── 10·(−cost) + 2·semantic + 1·lm
                              └─► corrected sentence + tree + edit script
```

Weights are ordered so syntax dominates by construction: one re-inflection
(0.4 × 10 = 4.0) outweighs the entire range of the other two terms combined
(2.0 + 1.0). Semantics only ever decides between repairs the parser already
considers equally good — a selectional restriction is a soft preference, not a
grammaticality judgement.

## Cost model

| Operation | Meaning | Cost |
|---|---|---|
| MATCH | token already carries the required tag | 0.0 |
| INFLECT | another inflection of the same lemma | 0.4 |
| INSERT / DELETE (function word) | a determiner, preposition, modal… | 0.6 |
| SUBSTITUTE | a word of a different lemma | 1.0 |
| INSERT / DELETE (content word) | invent or discard a noun, verb, adjective | 1.6 |

The open/closed word-class split is not cosmetic. With flat costs, deleting a
noun and inserting a determiner tie — so the parser "repairs"
`the dog ate pizza` by deleting `pizza`.

## Project structure

```
sg-ecep/
├── sgecep/                  # the parser package
│   ├── grammar.py           CFG + lexicon loading, coverage checking
│   ├── earley.py            baseline Earley (the evaluation's control condition)
│   ├── ec_earley.py         error-correcting chart, k-best repair enumeration  ← core
│   ├── edits.py             edit operations and the cost model
│   ├── trees.py             parse trees, slots, rendering
│   ├── lesk.py              Lesk WSD with most-frequent-sense interpolation
│   ├── semantics.py         shallow SRL + selectional restriction checking
│   ├── lm.py                interpolated Kneser-Ney bigram (+ add-k baseline)
│   ├── generate.py          random CFG sampler, used to build the LM corpus
│   ├── rerank.py            slot filling and candidate ranking
│   ├── pipeline.py          end-to-end facade
│   └── cli.py               command-line interface
├── data/
│   ├── grammar.cfg                    70 rules, agreement + subcategorisation
│   ├── lexicon.tsv                    GENERATED — edit tools/build_lexicon.py
│   ├── semantic_types.json            coarse types ↔ WordNet anchors + fallback
│   ├── selectional_restrictions.json  verb → permitted argument types
│   └── corpus.txt                     GENERATED — python -m eval.build_corpus
├── eval/                    # corruption generator, evaluation runner, corpus builder
├── tests/                   # 78 pytest tests
├── demo/                    # sentences for `--demo`
├── docs/
│   ├── ALGORITHM.md         algorithm walkthrough
│   ├── PROJECT_PLAN.md      design plan and full references
│   ├── EVALUATION_RESULTS.txt
│   └── media/sgecep-demo.mp4
├── tools/
│   ├── build_lexicon.py     regenerates data/lexicon.tsv
│   └── setup_wordnet.py     downloads the NLTK WordNet data
├── pyproject.toml
└── requirements.txt
```

Two files are generated and should be rebuilt rather than hand-edited:

```bash
python tools/build_lexicon.py     # data/lexicon.tsv
python -m eval.build_corpus       # data/corpus.txt
```

## Notes from building it

Things that were not obvious in advance, and are worth knowing if you extend
this:

- **An over-permissive grammar is actively harmful.** The first CFG had no
  verb subcategorisation and random sampling produced `a patient on she ate
  very`. The parser only repairs as far as the nearest string the grammar
  accepts, so a loose grammar produces bad corrections. The grammar now
  encodes subcategorisation in the tag (`VTZ` = transitive 3sg present).
- **WordNet anchors must be read off actual hypernym paths.** `book.n.01`
  sits under `publication.n.01 → artifact.n.01` and **never** under
  `communication.n.02`. Anchors chosen by intuition silently misclassify every
  written work.
- **Proper nouns must bypass WordNet.** `john.n.01` is a lavatory.
- **Textbook Lesk underperforms most-frequent-sense** on short contexts; it
  resolved `cat` to a caterpillar-tractor sense. The implementation
  interpolates a frequency prior, and `mfs_weight=0` recovers textbook Lesk.

## Limitations

- The grammar is small and hand-written. Scaling would need a PCFG induced
  from a treebank.
- The instrument check is coarse: `saw the man with the hammer` passes,
  because WordNet has no *used-for* relation. ConceptNet would be needed.
- Minimum edit distance is not writer intent. The cheapest repair is sometimes
  not the intended one, which is what caps oracle recovery near 60%.
- Lesk remains weak on short contexts even with the frequency prior.

## References

Earley (1970) · Aho & Peterson (1972) · Lyon (1974) · Lesk (1986) ·
Banerjee & Pedersen (2002) · Kneser & Ney (1995) · Sakaguchi et al. (2017).
Full citations in `docs/PROJECT_PLAN.md`.
