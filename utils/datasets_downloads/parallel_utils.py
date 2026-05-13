from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, Iterable, Iterator, TypeVar


T = TypeVar("T")
R = TypeVar("R")


def iter_threaded(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int,
    max_pending: int | None = None,
) -> Iterator[R]:
    max_workers = max(1, int(max_workers))
    if max_workers == 1:
        for item in items:
            yield fn(item)
        return

    max_pending = max(max_pending or max_workers * 2, max_workers)
    iterator = iter(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = set()

        def submit_one() -> bool:
            try:
                item = next(iterator)
            except StopIteration:
                return False
            pending.add(executor.submit(fn, item))
            return True

        for _ in range(max_pending):
            if not submit_one():
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                submit_one()
