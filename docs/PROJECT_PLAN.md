# Project Plan

**Semantically-Guided Error-Correcting Earley Parser (SG-ECEP)**
Natural Language Processing — course mini-project

---

## 1. Problem statement

A conventional parser is a *recogniser*: given a sentence and a grammar, it
answers yes or no. Faced with `the students reads the book` it simply fails,
and reports nothing useful about why, or about how close the sentence was.

This project builds a parser that repairs instead of rejecting. Given an
ungrammatical sentence it finds the **cheapest sequence of edits** that brings
the sentence inside the language the grammar defines, returns the corrected
sentence together with its parse tree and an explicit edit script, and — where
edit distance alone cannot decide — uses **semantic plausibility** to choose
between equally cheap repairs.

## 2. Why this topic

The faculty example (using an Earley parser to *generate* rather than
*predict*) points at an underlying idea: a grammar is a declarative object,
and parsing is only one way to use it. This project takes the other branch of
that idea — using the chart to search over *repairs* — and adds a component
the course covers but rarely connects to parsing: word sense disambiguation
and selectional restrictions.

The combination is what makes the project distinctive. Error-correcting
parsing is old (Aho & Peterson 1972; Lyon 1974) and selectional restrictions
are old, but using WSD-driven selectional restrictions to *break ties inside*
a minimum-distance parser's repair search is not a standard student project,
and the implementation here departs from the classical algorithms in a way
that makes the semantic layer load-bearing rather than decorative (§4.2).

## 3. Syllabus coverage

| Module | Topic | Where it is used |
|---|---|---|
| 1 | Tokenisation, minimum edit distance | `edits.py`; the whole repair search is edit distance over a grammar |
| 2 | N-grams, smoothing, perplexity | `lm.py` — interpolated Kneser-Ney bigram, add-k baseline |
| 3 | CFGs, Earley parsing | `earley.py`, `ec_earley.py` |
| 3 | Parse trees, ambiguity | `trees.py`; PP-attachment ambiguity |
| 3 | Lesk, WSD | `lesk.py` |
| 3 | Semantic role labelling | `semantics.py` — shallow rule-based SRL |
| 3 | WordNet, semantic relations | `semantics.py` — hypernym-closure type checking |

Several distinct syllabus topics in one working system, rather than one topic
demonstrated in isolation.

## 4. Technical approach

### 4.1 Error-correcting Earley

The standard Earley chart is extended with a cost and four edit operations.
An item's cost is the cost of the edits inside its own span (the usual inside
convention), so COMPLETE adds the costs of the two items it joins. Costs are
non-negative, so the chart is relaxed with a priority queue.

| Operation | Meaning | Default cost |
|---|---|---|
| MATCH | token already carries the required tag | 0.0 |
| INFLECT | swap for another inflection of the same lemma | 0.4 |
| INSERT (function word) | supply a missing determiner, preposition… | 0.6 |
| DELETE (function word) | drop a spurious determiner | 0.6 |
| SUBSTITUTE | swap for a word of a different lemma | 1.0 |
| INSERT / DELETE (content word) | invent or discard a noun, verb, adjective | 1.6 |

The open/closed word-class split is not cosmetic. With flat costs, deleting
the noun and inserting a determiner tie, so the parser "repairs"
`the dog ate pizza` by deleting `pizza`.

Termination: INSERT is the only operation that consumes no input, and its cost
is strictly positive, so a finite cost budget bounds consecutive insertions.

### 4.2 The departure from the classical algorithm

Aho–Peterson and Lyon produce repaired *strings*. This parser produces repair
**templates**: when it substitutes or inserts, it records only the
pre-terminal the grammar demanded — a *slot* — and does not commit to a word.

This matters twice over. The chart never branches over the vocabulary, so it
stays small. And it is what gives the semantic layer real work: filling a noun
slot is a choice among sixty-odd nouns that are **identical in edit cost**, so
edit distance cannot decide it at all. The semantics is not re-ranking
finished output; it is choosing content.

### 4.3 Semantic layer

1. **Shallow SRL** reads predicate-argument structure (ARG0/ARG1/ARG2) off the
   parse tree using structural rules, which is possible because the grammar's
   verb frames are known.
2. **Lesk WSD** picks a sense for each argument head, interpolated with
   WordNet's most-frequent-sense prior.
3. **Selectional restrictions** test the chosen sense's hypernym closure
   against the types the verb permits in that role.

Committing to one sense first is what gives the checker teeth. A checker that
accepted if *any* sense passed would let `the cat ate the homework` through,
because some sense of almost any noun can be coerced into almost any type.

### 4.4 Reranking

```
total = 10.0 · (−syntactic_cost) + 2.0 · semantic_score + 1.0 · lm_score
```

Weights are ordered so syntax dominates by construction: one re-inflection
(0.4 × 10 = 4.0) outweighs the entire range of the other two terms combined
(2.0 + 1.0). Semantics only ever decides between repairs the parser already
considers equally good. This is the right division of labour — a selectional
restriction is a soft preference, not a grammaticality judgement, and letting
it override the parser would mean rejecting grammatical sentences for being
surprising.

## 5. Deliverables

- Working system with CLI (`python -m sgecep.cli`)
- Reproducible evaluation harness with ablations
- 78-test suite, runnable without WordNet
- Report covering algorithm, evaluation and limitations

## 6. Timeline (6 weeks)

| Week | Work | Output |
|---|---|---|
| 1 | Literature survey (Earley 1970; Aho & Peterson 1972; Lyon 1974; Lesk 1986; Banerjee & Pedersen 2002). Grammar and lexicon design. | Survey notes, `grammar.cfg`, `lexicon.tsv` |
| 2 | Baseline Earley parser; parse-tree extraction; chart dump for the report. | `earley.py`, `trees.py`, tests |
| 3 | Error-correcting extension: weighted chart, four operations, k-best enumeration. | `ec_earley.py`, `edits.py` |
| 4 | Lesk WSD, shallow SRL, selectional restrictions, semantic type index. | `lesk.py`, `semantics.py`, data files |
| 5 | Language model, reranker, pipeline, CLI, corpus generation. | `lm.py`, `rerank.py`, `pipeline.py`, `cli.py` |
| 6 | Evaluation harness, ablation study, error analysis, report, demo. | `eval/`, report, slides |

## 7. Evaluation design

Grammatical sentences are corrupted on purpose and the system is measured on
how often it recovers the original. Five corruption types mirror the edit
operations: `agreement`, `drop_word`, `duplicate`, `swap_word`, `swap_pos`.

Three metrics, over the genuinely ungrammatical cases only:

- **Exact recovery** — top correction equals the gold sentence
- **Oracle recovery** — gold sentence appears anywhere in the candidate list
- **Semantic validity** — top correction satisfies every applicable restriction

Restricting to ungrammatical cases matters: roughly a third of corruptions
yield another perfectly grammatical sentence, and the system is *right* to
leave those alone. Scoring them as failures would measure the wrong thing.

Four configurations: `baseline` (plain Earley), `syntax-only`, `syntax+lm`,
`full`.

### Measured outcome

695 ungrammatical cases from 200 gold sentences:

| configuration | exact | oracle | semantically valid |
|---|---|---|---|
| baseline | 0.0% | 0.0% | 0.0% |
| syntax-only | 41.2% | 51.2% | 87.5% |
| syntax + LM | 47.1% | 59.4% | 91.2% |
| full | 47.2% | 59.7% | 93.5% |

By corruption type (exact recovery, `full`): agreement 97.6%, duplicate 43.6%,
swap_pos 34.2%, swap_word 25.0%, drop_word 12.7%.

The semantic layer moves exact recovery by 0.1 points and semantic validity by
6.0. The report should treat this as the headline finding rather than bury it:
semantics cannot identify which of sixty equally-licensed nouns a writer
originally chose, but it can keep the correction plausible.

The parser returned more than one minimum-cost repair in 304 of the 695 cases
(44%), which is the direct evidence that tie-breaking is a real problem rather
than a contrived one.

## 8. Known limitations (to be stated in the report)

- The grammar is small and hand-written; scaling would need a PCFG induced
  from a treebank.
- The instrument check is coarse: `saw the man with the hammer` passes,
  because WordNet has no *used-for* relation. ConceptNet would be needed.
- Minimum edit distance is not the same as writer intent; the cheapest repair
  is sometimes not the intended one, which caps oracle recovery.
- Lesk remains weak on short contexts even with the MFS prior.

## 9. References

1. Earley, J. (1970). An efficient context-free parsing algorithm. *CACM* 13(2).
2. Aho, A. V. & Peterson, T. G. (1972). A minimum distance error-correcting parser for context-free languages. *SIAM J. Computing* 1(4).
3. Lyon, G. (1974). Syntax-directed least-errors analysis for context-free languages. *CACM* 17(1).
4. Lesk, M. (1986). Automatic sense disambiguation using machine readable dictionaries. *SIGDOC*.
5. Banerjee, S. & Pedersen, T. (2002). An adapted Lesk algorithm for word sense disambiguation using WordNet. *CICLing*.
6. Kneser, R. & Ney, H. (1995). Improved backing-off for m-gram language modeling. *ICASSP*.
7. Jurafsky, D. & Martin, J. H. *Speech and Language Processing*, 3rd ed. draft.
8. Sakaguchi, K. et al. (2017). Error-repair dependency parsing for ungrammatical texts. *ACL*.
