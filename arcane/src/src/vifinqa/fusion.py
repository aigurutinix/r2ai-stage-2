"""Turn cross-encoder scores into the ordering that ships.

Every replacement tried on this task has lost and every fusion that earned its
place has won, so the reranker enters as one more reciprocal rank rather than as
the new order. Lives in the package rather than beside the CLI because
The rule lives in the package so every CLI and notebook uses the same behavior.

The fixed equal weight is part of the published retrieval contract; it is not
a tunable parameter.
"""

from collections import defaultdict
from pathlib import Path

from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import tokenize


RRF_OFFSET = 60


def reciprocal(rank: int | None) -> float:
    return 1.0 / (RRF_OFFSET + rank) if rank else 0.0


def fuse_orders(orders: list[list[str]]) -> list[str]:
    """Fuse independent table orders with equal reciprocal-rank contributions."""
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    for order in orders:
        for rank, table_id in enumerate(order, 1):
            scores[table_id] += reciprocal(rank)
            first_seen.setdefault(table_id, len(first_seen))
    return sorted(scores, key=lambda table_id: (-scores[table_id], first_seen[table_id]))


def prune_line_item_tables(
    record: dict, order: list[str], line_items: list[str], extra_nonmatches: int = 2,
) -> list[str]:
    """Keep exact line-item carriers, plus a small recall cushion."""
    if not line_items or extra_nonmatches < 0:
        return order
    candidates = {candidate["table_id"]: candidate for candidate in record["candidates"]}
    normalized_items = [" ".join(tokenize(item)) for item in line_items]
    matches, nonmatches = [], []
    for table_id in order:
        candidate = candidates.get(table_id)
        text = " ".join(tokenize(candidate.get("text", ""))) if candidate else ""
        (matches if any(f" {item} " in f" {text} " for item in normalized_items)
         else nonmatches).append(table_id)
    return order if not matches else matches + nonmatches[:extra_nonmatches]


def first_stage(candidate: dict, arm: str) -> float:
    """What the retrievers said about a candidate, before the reranker ran.

    Three arms over one scored pool, which is the only comparison available
    without a second GPU run — and a second run was measured to differ from the
    first by more than most effects worth testing.

    `hybrid` sums the two contributions rather than averaging them, so a table
    both retrievers returned outranks one that only either did. That agreement
    bonus is what reciprocal rank fusion exists for; averaging would throw it
    away.
    """
    sparse, dense = candidate.get("sparse_rank"), candidate.get("dense_rank")
    if arm == "sparse":
        return reciprocal(sparse)
    if arm == "dense":
        return reciprocal(dense)
    return reciprocal(sparse) + reciprocal(dense)


def fuse(
    candidates: list[dict], scores: dict[str, float], mode: str, weight: float = 0.5,
    arm: str = "sparse",
) -> list[str]:
    """Order one question's candidates by weighted reciprocal rank.

    `weight` is the share given to the retrievers: 1.0 retrieval alone, 0.0 the
    model alone, 0.5 the unweighted sum every submission has used.

    `arm` selects which retrievers a candidate has to have been returned by. A
    candidate the chosen arm never produced is not ranked at all, because a
    candidate list is a claim about what the first stage found and borrowing the
    other arm's candidates would not be that arm's system.
    """
    retrieved = {candidate["table_id"]: first_stage(candidate, arm)
                 for candidate in candidates if first_stage(candidate, arm) > 0.0}
    ranked_by_model = sorted(
        (table_id for table_id in retrieved if table_id in scores),
        key=lambda table_id: (-scores[table_id], -retrieved[table_id]),
    )
    model_rank = {table_id: rank for rank, table_id in enumerate(ranked_by_model, 1)}
    if mode == "replace":
        return ranked_by_model + [t for t in sorted(retrieved, key=lambda i: -retrieved[i])
                                  if t not in model_rank]
    return sorted(
        retrieved,
        key=lambda table_id: (
            -(weight * retrieved[table_id]
              + (1 - weight) * reciprocal(model_rank.get(table_id))),
            -retrieved[table_id],
        ),
    )


def load_scores(paths: list[Path]) -> dict[int, dict[str, float]]:
    """Merge score files, inside the per-question dict as well as across it.

    Sharded runs never share a question, but a depth extension does and differs
    only in which candidates it judged, so the union has to reach inside.
    """
    scored: dict[int, dict[str, float]] = defaultdict(dict)
    for path in paths:
        for record in load_jsonl(path):
            scored[record["id"]].update(record["scores"])
    return scored


def rankings(
    pairs: dict, scored: dict, mode: str, weight: float, arm: str = "sparse",
) -> dict[str, list[str]]:
    """The shipped ordering rule, keyed by question ID as ranking.json is."""
    ranking = {}
    for identifier, record in pairs.items():
        # Without model scores the arm is still an ordering: its retrievers'
        # reciprocal ranks, which for the hybrid arm is the fusion of the two.
        # Falling back to the file's order would silently rank the dense arm by
        # whatever order the exporter happened to write.
        order = [table_id for table_id, _ in sorted(
            ((candidate["table_id"], first_stage(candidate, arm))
             for candidate in record["candidates"] if first_stage(candidate, arm) > 0.0),
            key=lambda pair: -pair[1],
        )]
        if identifier in scored:
            order = fuse(record["candidates"], scored[identifier], mode, weight, arm)
        ranking[str(identifier)] = order
    return ranking
