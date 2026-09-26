"""Correctness-gated benchmark for mojo-arrayfire.

Where a baseline exists in the real package, it is the real package: the
reference column for the index and length cases is the upstream Python code
executed verbatim by `tests/_upstream.py`, one call per element, which is how
ArrayFire's binding actually builds an index record. For the marshalling cases
the fastest reasonable NumPy formulation is the reference.

Every case checks its result before timing.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "tests"))

import mojo_arrayfire as maf  # noqa: E402
import _upstream  # noqa: E402

_up_seq = _upstream.seq_triple
_up_length = _upstream._slice_to_length()


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _random_slices(rng, n, dim):
    out = []
    for _ in range(n):
        kind = rng.integers(0, 4)
        if kind == 0:
            out.append(slice(None))
        elif kind == 1:
            a, b = sorted(rng.integers(-dim, dim, size=2))
            out.append(slice(int(a), int(b)))
        elif kind == 2:
            a, b = sorted(rng.integers(0, dim, size=2))
            step = int(rng.choice([1, 2, 3]))
            out.append(slice(int(a), int(b), step))
        else:
            a, b = sorted(rng.integers(-dim, dim, size=2))
            out.append(slice(int(a), int(b), -int(rng.choice([1, 2]))))
    return out


def bench_seq(n: int = 100_000, dim: int = 4096):
    rng = np.random.default_rng(0)
    slices = _random_slices(rng, n, dim)
    got = maf.seq_batch(slices)
    for i in (0, 1, n // 2, n - 1):
        assert tuple(got[i]) == _up_seq(slices[i]), f"slice {i}"

    def run_mine():
        maf.seq_batch(slices)

    def run_theirs():
        for s in slices:
            _up_seq(s)

    return f"Seq batch n={n}", _time(run_theirs, 3), _time(run_mine, 3)


def bench_slice_length(n: int = 100_000, dim: int = 4096):
    rng = np.random.default_rng(1)
    slices = _random_slices(rng, n, dim)
    got = maf.slice_lengths(slices, [dim] * n)
    for i in (0, 1, n // 2, n - 1):
        assert got[i] == _up_length(slices[i], dim), f"slice {i}"

    def run_mine():
        maf.slice_lengths(slices, [dim] * n)

    def run_theirs():
        for s in slices:
            _up_length(s, dim)

    return f"slice lengths n={n}", _time(run_theirs, 3), _time(run_mine, 3)


def bench_reorder(n0: int = 2048, n1: int = 2048, dtype=np.float64):
    rng = np.random.default_rng(2)
    src = rng.standard_normal((n0, n1)).astype(dtype)
    got = maf.reorder2d(src)
    assert np.array_equal(got, np.asfortranarray(src)), "reorder2d mismatch"
    return (
        f"reorder2d {n0}x{n1} {np.dtype(dtype).name}",
        _time(lambda: np.asfortranarray(src)),
        _time(lambda: maf.reorder2d(src)),
    )


def bench_broadcast_index(out_dims=(512, 512, 4, 1), in_dims=(1, 512, 1, 1)):
    total = int(np.prod(out_dims))
    src = np.arange(int(np.prod(in_dims)), dtype=np.int64)
    got = maf.broadcast_index(out_dims, in_dims)
    expect = np.broadcast_to(
        src.reshape(in_dims, order="F"), out_dims
    ).reshape(-1, order="F")
    assert got.tolist() == [int(v) for v in expect], "broadcast_index mismatch"
    return (
        f"broadcast_index {total}",
        _time(lambda: np.broadcast_to(src.reshape(in_dims, order="F"), out_dims)
              .reshape(-1, order="F")),
        _time(lambda: maf.broadcast_index(out_dims, in_dims)),
    )


def bench_broadcast_shape(n: int = 50_000):
    rng = np.random.default_rng(3)
    groups = []
    for _ in range(n):
        a = int(rng.integers(1, 64))
        b = int(rng.integers(1, 64))
        groups.append(((a, b, 1, 1), (a, 1, 1, 1)))
    got = maf.broadcast_shape(groups)
    assert got[0].tolist() == list(np.broadcast_shapes(*groups[0]))

    def run_mine():
        maf.broadcast_shape(groups)

    def run_theirs():
        for g in groups:
            np.broadcast_shapes(*g)

    return f"broadcast_shape n={n}", _time(run_theirs, 3), _time(run_mine, 3)


def bench_strides(reps: int = 200_000):
    dims = (7, 11, 13, 17)

    def run_mine():
        for _ in range(reps):
            maf.strides4(dims)

    def run_theirs():
        d0, d1, d2, _ = dims
        for _ in range(reps):
            (1, d0, d0 * d1, d0 * d1 * d2)

    return f"strides4 x{reps}", _time(run_theirs, 3), _time(run_mine, 3)


def main():
    print(f"{'case':<36}{'reference':>12}{'mojo-arrayfire':>17}{'ratio':>10}")
    print("-" * 76)
    for fn in (
        bench_seq,
        bench_slice_length,
        bench_reorder,
        bench_broadcast_index,
        bench_broadcast_shape,
        bench_strides,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<36}{ref*1e3:>10.2f}ms{got*1e3:>15.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
