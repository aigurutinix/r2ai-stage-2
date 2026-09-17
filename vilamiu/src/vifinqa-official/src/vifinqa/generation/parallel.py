"""Generate questions for multiple candidates concurrently with a thread pool.

All difficulty tiers previously used the same sequential pattern: try candidates one
at a time and stop once the requested count is reached. The OpenAI Python client is
safe for independent concurrent calls, so threads avoid making ``ChatLLM`` async.

Each successful question is checkpointed immediately through the thread-safe
``JsonlWriter``; failed or incomplete attempts need no separate checkpoint file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import islice
from threading import Lock
from typing import TypeVar

from vifinqa.generation.budget import GenerationBudgetExceeded
from vifinqa.generation.schemas import QARecord
from vifinqa.generation.output.writer import JsonlWriter

logger = logging.getLogger(__name__)

T = TypeVar("T")

# A candidate can make several LLM calls. Sequential execution is the default so
# ``--count 1`` really processes only one candidate; batch runs may raise
# ``--max-workers`` explicitly.
DEFAULT_MAX_WORKERS = 1


def run_parallel_generation(
    *,
    work_items: Iterable[T],
    build_record: Callable[[T], QARecord | None],
    count: int,
    writer: JsonlWriter,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """Run ``build_record`` concurrently over work items.

    Keep at most ``max_workers`` futures active and write successful records until
    ``count`` is reached. Ignore futures that finish after the quota is met. A failure
    in one candidate is logged and treated as ``None`` without failing the batch.
    """
    generated = 0
    lock = Lock()
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be at least 1 or None")
    items_iter = iter(islice(work_items, max_candidates)) if max_candidates is not None else iter(work_items)
    in_flight: set[Future] = set()

    def _submit_next(executor: ThreadPoolExecutor) -> bool:
        if should_stop is not None and should_stop():
            return False
        try:
            item = next(items_iter)
        except StopIteration:
            return False
        in_flight.add(executor.submit(build_record, item))
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_workers):
            if not _submit_next(executor):
                break

        while in_flight:
            done, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.discard(future)
                try:
                    record = future.result()
                except GenerationBudgetExceeded as exc:
                    logger.info("Stopping generation because the budget is exhausted: %s", exc)
                    record = None
                except Exception:
                    logger.exception("Failed to generate a candidate; skipping it.")
                    record = None

                with lock:
                    if record is not None and generated < count:
                        writer.append(record)
                        generated += 1
                        logger.info("Generated %d/%d", generated, count)

                if generated < count and not (should_stop is not None and should_stop()):
                    _submit_next(executor)

    return generated
