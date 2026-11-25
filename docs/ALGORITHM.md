# The Algorithm

A worked explanation of the error-correcting Earley parser, intended as the
basis for the report's methodology section.

---

## 1. Recap: standard Earley

An Earley **item** is a dotted rule plus the input position where it started:

```
A -> α • β , i        "we are parsing an A that began at position i;
                       α is matched, β is still to come"
```

The chart has one cell per input position. Three operations fill it:

- **PREDICT** — item `A -> α • B β` at position `j` adds `B -> • γ` at `j`
  for every rule `B -> γ`.
- **SCAN** — item `A -> α • a β` at `j`, where `a` matches input token `j`,
  adds `A -> α a • β` at `j+1`.
- **COMPLETE** — item `B -> γ •` at `j` with origin `i` advances every item at
  `i` that was waiting on `B`.

The sentence is accepted if `S -> γ •` with origin 0 appears at position *n*.

Earley handles left recursion natively (`VPS -> VPS ADVP` here) and runs in
O(n³) generally, O(n²) for unambiguous grammars.

### Nullable non-terminals

A predicted nullable non-terminal must be completed immediately, or items
waiting on it never advance — the classic epsilon bug. `Grammar.nullable()`
computes the nullable set by fixpoint and `earley.py` applies the fix.

## 2. Adding edits

Four operations repair the *input* so that it fits the grammar.

| Operation | Effect on the chart | Consumes input? |
|---|---|---|
| MATCH | `j → j+1`, dot advances, cost 0 | yes |
| SUBSTITUTE / INFLECT | `j → j+1`, dot advances, cost > 0 | yes |
| DELETE | `j → j+1`, **dot does not move** | yes |
| INSERT | stays at `j`, dot advances | **no** |

DELETE not moving the dot is the key subtlety: the item's span grows to cover
the dropped token while its progress through the rule is unchanged. It
therefore applies to *every* item at a position, not only to those expecting a
terminal.

### Weighting

Each item carries the cost of the edits **inside its own span** (the inside
convention). So:

```
PREDICT    new item starts at cost 0
SCAN       cost(child) = cost(parent) + edit_cost
INSERT     cost(child) = cost(parent) + insert_cost
DELETE     cost(child) = cost(parent) + delete_cost
COMPLETE   cost(result) = cost(waiting_item) + cost(completed_item)
```

All costs are non-negative, so each chart position is relaxed with a priority
queue, with items re-queued whenever a cheaper derivation is found.

### Termination

INSERT is the only operation that does not consume input, so it is the only
possible source of non-termination. Its cost is strictly positive and the
total cost is capped by `max_cost`, which bounds consecutive insertions at
`⌈max_cost / insert_cost⌉`. `CostModel.validate()` enforces the positivity
precondition rather than leaving it implicit.

## 3. Worked example

Input: `the students reads the book`, which the grammar rejects because `NPP`
(plural) requires `VPP`, and `reads` is `VTZ` (3sg).

Two repairs tie at cost **0.40**, one INFLECT each:

| Repair | Edit | Result |
|---|---|---|
| fix the subject | `students → <NNS>` | `the student reads the book` |
| fix the verb | `reads → <VTP>` | `the students read the book` |

Both are minimum-cost, both are grammatical, and **edit distance cannot
separate them**. This tie is not a flaw in the cost model; it is a genuine
ambiguity in the input, and it is exactly why the reranker exists. A test
asserts the tie rather than relying on it (`test_agreement_error_yields_a_genuine_tie`).

### Why slots

Note the repair outputs above contain `<NNS>` and `<VTP>`, not words. The
parser records only the *tag* required, deferring word choice:

```
the <NNS> reads the book      ← slot: any of ~60 singular nouns
the students <VTP> the book   ← slot: any non-3sg present transitive verb
```

For the INFLECT case the reranker constrains the slot to the original word's
lemma, so `students → student` is forced. For a SUBSTITUTE or INSERT slot
there is no such constraint, and the choice falls entirely to semantics and
the language model.

## 4. Recovering repairs from the chart

The chart stores, for each item, its best cost **and every incoming edge**
within budget. Repairs are enumerated by walking that edge DAG bottom-up with
memoisation, pruning branches that exceed the budget.

Edges are visited in order of an **admissible lower bound**:

```
lower_bound(edge) = edge.cost
                  + best_cost(edge.prev)
                  + best_cost(edge.child)
```

Ordering by the edge's own cost instead is wrong, and was a real bug during
development: a zero-cost COMPLETE edge whose child is expensive gets explored
before a cheap INSERT, and the per-item beam then discards the cheapest
derivation. The symptom was diagnostic — `best_cost()` reported 0.6 while
`repairs()` could not produce any derivation at that cost. A regression test
now asserts the two agree on every example
(`test_enumeration_reaches_the_chart_minimum`).

The per-item beam makes enumeration complete with respect to **cost** (the
cheapest repair is always found) but only approximately complete with respect
to the full k-best list — the right trade when a reranker only inspects a
handful of candidates.

## 5. Complexity

Let *n* be sentence length, *|G|* the grammar size, *C* the cost budget and
*c* the cheapest positive edit cost.

- Items per position: O(|G| · n), as in standard Earley.
- DELETE copies a position forward: O(items) extra per position.
- INSERT adds at most ⌈C/c⌉ dot-advances per item within a position.
- Overall: **O(n³ · |G| · C/c)** — standard Earley multiplied by the budget.

In practice the budget is small (3.0 with a cheapest edit of 0.4) and
sentences are short, so parsing is a few tens of milliseconds.

## 6. Semantic layer

```
parse tree
   │
   ├─ shallow SRL ──────► predicate(ARG0, ARG1, ARG2) + instrument
   │
   ├─ Lesk WSD ─────────► one WordNet sense per argument head
   │
   └─ selectional check ► sense's hypernym closure ∩ verb's permitted types
                          │
                          └─► score = satisfied / checked, or 0.5 if nothing
                              could be checked
```

The neutral 0.5 is deliberate: absence of evidence must not look like evidence
of plausibility, nor of implausibility. A verb with no recorded restrictions
scores neutral, never penalised.

### Why the MFS prior

Textbook Lesk scores each sense by gloss overlap with the context. On short
contexts the overlap is usually zero for every sense, and a single coincidental
word promotes a rare sense — during development it resolved `cat` to a
caterpillar-tractor sense. The implementation scores

```
overlap + mfs_weight / (sense_rank + 1)
```

so a rare sense needs roughly two more overlapping words than sense 1 to win.
Setting `mfs_weight=0` recovers textbook Lesk, and a test asserts that.

### A WordNet trap worth reporting

Anchoring coarse types to WordNet synsets must be done against **actual
hypernym paths**, not intuition. `book.n.01` sits under
`publication.n.01 → work.n.02 → creation.n.02 → artifact.n.01` and **never**
under `communication.n.02`. An anchor set chosen by intuition silently
misclassifies every written work. Similarly, proper nouns must bypass WordNet
entirely: `john.n.01` is a lavatory and `mark.n.01` a blemish, and Lesk has no
way to know the token is a name.

## 7. Reranking

```
total = w_syn · (−cost) + w_sem · semantic + w_lm · lm_normalised
      =  10.0 · (−cost) +   2.0 · semantic +  1.0 · lm_normalised
```

The LM term is min-max normalised across the candidate set, so the weights
mean the same thing regardless of corpus or sentence length.

Because `10.0 × 0.4 = 4.0 > 2.0 + 1.0`, a repair one re-inflection cheaper can
never be overturned by the other two signals combined. `test_weights_keep_syntax_dominant`
asserts this invariant directly, so tuning the weights cannot silently break it.
