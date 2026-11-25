"""Error-correcting Earley parser.

Extends the Earley chart of ``earley.py`` with the four edit operations in
``edits.py``.  The descent is from Aho & Peterson's minimum-distance
error-correcting parser (1972) and Lyon's Earley-based formulation (1974);
the implementation here is written from the algorithm rather than ported from
either, and differs from both in one respect that matters downstream.

**Repair templates, not repaired strings.**  When the parser substitutes or
inserts a token it does *not* commit to a particular word.  It records only
the pre-terminal the grammar demanded -- a *slot*.  A single chart therefore
encodes a small set of repair skeletons, and choosing the words that fill the
slots is left to the semantic layer (``semantics.py``) and the language model
(``lm.py``).  This keeps the chart small (no branching over the vocabulary)
and, more importantly, it is the point at which semantics gets to influence
the repair instead of merely re-ranking finished strings.

Weighting follows the usual inside-cost convention: an item's cost is the cost
of the edits strictly inside its own span.  COMPLETE therefore adds the costs
of the two items it joins, while PREDICT starts a fresh item at zero.  Costs
are non-negative, so the chart is relaxed with a priority queue; items are
re-queued whenever a cheaper derivation of them is found.

Termination rests on two facts: INSERT is the only operation that does not
consume input, and its cost is strictly positive, so a bounded cost budget
bounds the number of consecutive insertions.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

from .edits import CostModel, Edit, EditKind
from .grammar import Grammar, Lexicon, Rule
from .trees import Tree

# (chart position, rule index, dot, origin)
ItemKey = tuple[int, int, int, int]

SLOT = "\u25a1"  # placeholder shown for an unfilled slot: □


@dataclass(frozen=True)
class Edge:
    """How one item was derived from its predecessors."""

    kind: str                      # init | scan | insert | delete | complete
    cost: float                    # cost contributed by THIS edge alone
    prev: ItemKey | None = None    # the same rule with the dot one step back
    child: ItemKey | None = None   # completed constituent (kind == complete)
    tag: str | None = None         # pre-terminal consumed (scan / insert)
    token_index: int | None = None  # index into the original tokens
    original: str | None = None    # original token at that index
    edit_kind: EditKind | None = None


@dataclass
class ItemEntry:
    best_cost: float
    edges: list[Edge] = field(default_factory=list)
    edge_set: set[Edge] = field(default_factory=set)


@dataclass
class Repair:
    """One candidate repair of the input.

    ``tree`` may contain slot leaves (``word is None``); ``edits`` lists every
    non-trivial operation.  ``syntactic_cost`` is the chart cost and is the
    only score the parser itself assigns -- semantic and LM scores are added
    later by the reranker.
    """

    tree: Tree
    edits: list[Edit]
    syntactic_cost: float

    def slot_count(self) -> int:
        return sum(1 for leaf in self.tree.leaves() if leaf.word is None)

    def surface(self) -> list[str]:
        """Token sequence with unfilled slots shown as ``<TAG>``."""
        return [
            leaf.word if leaf.word is not None else f"<{leaf.label}>"
            for leaf in self.tree.leaves()
        ]

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{' '.join(self.surface())}  (cost {self.syntactic_cost:.2f})"


class ErrorCorrectingEarleyParser:
    """Earley parser that repairs ungrammatical input as it parses."""

    def __init__(
        self,
        grammar: Grammar,
        lexicon: Lexicon,
        costs: CostModel | None = None,
        max_cost: float = 2.8,
        beam: int = 24,
    ) -> None:
        self.grammar = grammar
        self.lexicon = lexicon
        self.costs = costs or CostModel()
        self.costs.validate()
        self.max_cost = max_cost
        self.beam = beam

        # lemma -> tags, so INFLECT can be distinguished from SUBSTITUTE while
        # the chart is being built rather than after the fact.
        self._tags_by_lemma: dict[str, set[str]] = defaultdict(set)
        for entry in lexicon.entries:
            self._tags_by_lemma[entry.lemma].add(entry.tag)

    # -- cost model helpers -------------------------------------------------

    def _scan_edit(self, token: str, tag: str) -> tuple[EditKind, float]:
        """Classify consuming ``token`` as an instance of ``tag``."""
        if tag in self.lexicon.tags(token):
            return EditKind.MATCH, self.costs.match
        lemma = self.lexicon.lemma(token)
        if tag in self._tags_by_lemma.get(lemma, set()):
            return EditKind.INFLECT, self.costs.inflect
        return EditKind.SUBSTITUTE, self.costs.substitute

    # -- chart construction -------------------------------------------------

    def build_chart(self, tokens: list[str]) -> dict[ItemKey, ItemEntry]:
        n = len(tokens)
        chart: dict[ItemKey, ItemEntry] = {}
        # Items grouped by chart position, needed by DELETE (which copies a
        # whole position forward) and by COMPLETE (which scans the origin).
        by_position: list[list[ItemKey]] = [[] for _ in range(n + 1)]
        waiting: dict[tuple[int, str], list[ItemKey]] = defaultdict(list)

        def relax(key: ItemKey, cost: float, edge: Edge) -> bool:
            """Record ``edge`` for ``key``; return True if its cost improved."""
            if cost > self.max_cost + 1e-9:
                return False
            entry = chart.get(key)
            if entry is None:
                chart[key] = ItemEntry(best_cost=cost, edges=[edge], edge_set={edge})
                by_position[key[0]].append(key)
                _index(key)
                return True
            if edge not in entry.edge_set:
                entry.edges.append(edge)
                entry.edge_set.add(edge)
            if cost < entry.best_cost - 1e-12:
                entry.best_cost = cost
                return True
            return False

        def _index(key: ItemKey) -> None:
            position, rule_index, dot, _origin = key
            rule = self.grammar.rules[rule_index]
            if dot < len(rule.rhs):
                waiting[(position, rule.rhs[dot])].append(key)

        for rule in self.grammar.rules_for(self.grammar.start):
            relax((0, rule.index, 0, 0), 0.0, Edge("init", 0.0))

        for position in range(n + 1):
            # ---- closure at this position: PREDICT / COMPLETE / INSERT -----
            queue: list[tuple[float, ItemKey]] = [
                (chart[k].best_cost, k) for k in by_position[position]
            ]
            heapq.heapify(queue)
            settled: dict[ItemKey, float] = {}

            while queue:
                cost, key = heapq.heappop(queue)
                if cost > chart[key].best_cost + 1e-12:
                    continue  # stale queue entry
                if key in settled and settled[key] <= cost + 1e-12:
                    continue
                settled[key] = cost

                _pos, rule_index, dot, origin = key
                rule = self.grammar.rules[rule_index]

                if dot >= len(rule.rhs):
                    # COMPLETE: advance everything that was waiting on rule.lhs
                    # at this item's origin.
                    for parent in list(waiting.get((origin, rule.lhs), ())):
                        parent_cost = chart[parent].best_cost
                        new_key = (position, parent[1], parent[2] + 1, parent[3])
                        edge = Edge("complete", 0.0, prev=parent, child=key)
                        if relax(new_key, parent_cost + cost, edge):
                            heapq.heappush(queue, (chart[new_key].best_cost, new_key))
                    continue

                symbol = rule.rhs[dot]

                if self.grammar.is_nonterminal(symbol):
                    for sub_rule in self.grammar.rules_for(symbol):
                        sub_key = (position, sub_rule.index, 0, position)
                        if relax(sub_key, 0.0, Edge("init", 0.0)):
                            heapq.heappush(queue, (chart[sub_key].best_cost, sub_key))
                    if symbol in self.grammar.nullable():
                        new_key = (position, rule_index, dot + 1, origin)
                        edge = Edge("complete", 0.0, prev=key, child=None)
                        if relax(new_key, cost, edge):
                            heapq.heappush(queue, (chart[new_key].best_cost, new_key))
                else:
                    # INSERT: the grammar wants a token of this tag and the
                    # input does not supply one.  No input is consumed, so this
                    # stays inside the current chart position.
                    new_key = (position, rule_index, dot + 1, origin)
                    insert_cost = self.costs.insert_cost(symbol)
                    edge = Edge(
                        "insert",
                        insert_cost,
                        prev=key,
                        tag=symbol,
                        token_index=position,
                        edit_kind=EditKind.INSERT,
                    )
                    if relax(new_key, cost + insert_cost, edge):
                        heapq.heappush(queue, (chart[new_key].best_cost, new_key))

            # ---- advance to the next position: SCAN / SUBSTITUTE / DELETE --
            if position >= n:
                continue
            token = tokens[position]
            delete_cost = self.costs.delete_cost(self.lexicon.tags(token))
            for key in list(by_position[position]):
                cost = chart[key].best_cost
                _pos, rule_index, dot, origin = key
                rule = self.grammar.rules[rule_index]

                # DELETE: the token is spurious.  The span grows but the dot
                # does not move, so this applies to every item at the position.
                relax(
                    (position + 1, rule_index, dot, origin),
                    cost + delete_cost,
                    Edge(
                        "delete",
                        delete_cost,
                        prev=key,
                        token_index=position,
                        original=token,
                        edit_kind=EditKind.DELETE,
                    ),
                )

                if dot >= len(rule.rhs):
                    continue
                symbol = rule.rhs[dot]
                if self.grammar.is_nonterminal(symbol):
                    continue

                edit_kind, edit_cost = self._scan_edit(token, symbol)
                relax(
                    (position + 1, rule_index, dot + 1, origin),
                    cost + edit_cost,
                    Edge(
                        "scan",
                        edit_cost,
                        prev=key,
                        tag=symbol,
                        token_index=position,
                        original=token,
                        edit_kind=edit_kind,
                    ),
                )

        return chart

    # -- acceptance ---------------------------------------------------------

    def accepting_keys(self, chart: dict[ItemKey, ItemEntry], n: int) -> list[ItemKey]:
        keys = []
        for rule in self.grammar.rules_for(self.grammar.start):
            key = (n, rule.index, len(rule.rhs), 0)
            if key in chart:
                keys.append(key)
        return sorted(keys, key=lambda k: chart[k].best_cost)

    # -- k-best repair enumeration -----------------------------------------

    def repairs(self, tokens: list[str], k: int = 8) -> list[Repair]:
        """Up to ``k`` distinct repair templates, cheapest first.

        Enumeration walks the edge DAG bottom-up with memoisation, pruning any
        branch whose cost already exceeds the budget.  Results per item are
        capped at ``beam`` so a highly ambiguous grammar cannot blow up: the
        parser stays complete with respect to *cost* (the cheapest repair is
        always found) while being only approximately complete with respect to
        the full k-best list, which is the right trade for a reranker that
        only ever looks at a handful of candidates.
        """
        chart = self.build_chart(tokens)
        n = len(tokens)
        accepting = self.accepting_keys(chart, n)
        if not accepting:
            return []

        memo_partial: dict[ItemKey, list[tuple[float, list[Tree], list[Edit]]]] = {}
        memo_constituent: dict[ItemKey, list[tuple[float, Tree, list[Edit]]]] = {}
        in_progress: set[ItemKey] = set()

        def partial(key: ItemKey) -> list[tuple[float, list[Tree], list[Edit]]]:
            """Derivations of a dotted item: the children built so far."""
            if key in memo_partial:
                return memo_partial[key]
            if key in in_progress:
                return []
            in_progress.add(key)
            results: list[tuple[float, list[Tree], list[Edit]]] = []

            def lower_bound(e: Edge) -> float:
                """Cheapest total this edge could possibly yield.

                Ordering edges by their own cost alone is wrong: a zero-cost
                COMPLETE edge whose child is expensive would be explored
                before a cheap INSERT, and the per-item beam would then
                discard the cheapest derivation.  Adding the children's
                already-known best costs makes the ordering admissible.
                """
                total = e.cost
                if e.prev is not None and e.prev in chart:
                    total += chart[e.prev].best_cost
                if e.child is not None and e.child in chart:
                    total += chart[e.child].best_cost
                return total

            for edge in sorted(chart[key].edges, key=lower_bound):
                if lower_bound(edge) > self.max_cost + 1e-9:
                    continue
                if edge.kind == "init":
                    results.append((0.0, [], []))
                    continue

                assert edge.prev is not None
                if edge.prev not in chart:
                    continue
                for prev_cost, prev_children, prev_edits in partial(edge.prev):
                    base = prev_cost + edge.cost
                    if base > self.max_cost + 1e-9:
                        continue

                    if edge.kind == "delete":
                        results.append(
                            (
                                base,
                                list(prev_children),
                                prev_edits
                                + [
                                    Edit(
                                        EditKind.DELETE,
                                        edge.token_index or 0,
                                        original=edge.original,
                                    )
                                ],
                            )
                        )
                    elif edge.kind in ("scan", "insert"):
                        assert edge.tag is not None
                        word = edge.original if edge.edit_kind is EditKind.MATCH else None
                        origin_label = {
                            EditKind.MATCH: "input",
                            EditKind.INFLECT: "substituted",
                            EditKind.SUBSTITUTE: "substituted",
                            EditKind.INSERT: "inserted",
                        }[edge.edit_kind or EditKind.MATCH]
                        leaf = Tree(edge.tag, word=word, origin=origin_label, terminal=True)
                        new_edits = list(prev_edits)
                        if edge.edit_kind is not EditKind.MATCH:
                            new_edits.append(
                                Edit(
                                    edge.edit_kind or EditKind.SUBSTITUTE,
                                    edge.token_index or 0,
                                    original=edge.original,
                                    tag=edge.tag,
                                )
                            )
                        results.append((base, prev_children + [leaf], new_edits))
                    elif edge.kind == "complete":
                        if edge.child is None:  # nullable non-terminal
                            results.append((base, list(prev_children), list(prev_edits)))
                            continue
                        for child_cost, child_tree, child_edits in constituent(edge.child):
                            total = base + child_cost
                            if total > self.max_cost + 1e-9:
                                continue
                            results.append(
                                (
                                    total,
                                    prev_children + [child_tree],
                                    prev_edits + child_edits,
                                )
                            )

                if len(results) > self.beam * 6:
                    break

            results.sort(key=lambda r: r[0])
            results = results[: self.beam]
            in_progress.discard(key)
            memo_partial[key] = results
            return results

        def constituent(key: ItemKey) -> list[tuple[float, Tree, list[Edit]]]:
            """Derivations of a *complete* item, wrapped into a subtree."""
            if key in memo_constituent:
                return memo_constituent[key]
            rule = self.grammar.rules[key[1]]
            built = [
                (cost, Tree(rule.lhs, children=list(children)), edits)
                for cost, children, edits in partial(key)
            ]
            built.sort(key=lambda r: r[0])
            built = built[: self.beam]
            memo_constituent[key] = built
            return built

        candidates: list[Repair] = []
        for key in accepting:
            for cost, tree, edits in constituent(key):
                candidates.append(Repair(tree=tree, edits=edits, syntactic_cost=cost))

        candidates.sort(key=lambda r: (r.syntactic_cost, len(r.edits)))
        return _dedupe(candidates, k)

    def best_cost(self, tokens: list[str]) -> float | None:
        chart = self.build_chart(tokens)
        accepting = self.accepting_keys(chart, len(tokens))
        if not accepting:
            return None
        return min(chart[key].best_cost for key in accepting)

    # -- diagnostics --------------------------------------------------------

    def chart_summary(self, tokens: list[str]) -> str:
        chart = self.build_chart(tokens)
        by_position: dict[int, int] = defaultdict(int)
        for key in chart:
            by_position[key[0]] += 1
        lines = [f"items: {len(chart)} across {len(by_position)} positions"]
        for position in sorted(by_position):
            marker = f" (token: {tokens[position]!r})" if position < len(tokens) else ""
            lines.append(f"  chart[{position}]: {by_position[position]} items{marker}")
        return "\n".join(lines)


def _dedupe(candidates: list[Repair], k: int) -> list[Repair]:
    """Collapse repairs that produce the same tagged skeleton."""
    seen: set[tuple] = set()
    out: list[Repair] = []
    for repair in candidates:
        signature = tuple(
            (leaf.label, leaf.word, leaf.origin) for leaf in repair.tree.leaves()
        )
        if signature in seen:
            continue
        seen.add(signature)
        out.append(repair)
        if len(out) >= k:
            break
    return out
