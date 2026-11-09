# VERIFY phase — mojo-arrayfire (Mojo port of Python package `arrayfire`)

Session date 2026-11-09. Assumption: README overstates until reproduced. No code
was written in this phase except to fix bugs shown by failing tests; all
benchmark figures below were measured in this session. The README table now
contains only numbers measured here (medians of the three runs below).

## Commands run in order

```
pixi run build && pixi run test    # exit 0, 76 passed in 0.67s (README said 59; collection is 76)
pixi run bench                     # exit 0; three runs, see §1
```

Environment: `pixi run bench` = `flock /tmp/mojo-bench.lock python bench/bench.py`,
best-of-3 per case inside `bench/bench.py` (`_time(..., 3)`), same process.
Upstream `arrayfire` C library is not importable here, so the "upstream"
column is the upstream Python source executed verbatim via `tests/_upstream.py`
(one call per element), as `bench/bench.py` and the README state.

## 1. What reproduced, what did not (per README row)

Three consecutive `pixi run bench` runs (same build, same process per run):

| case | run 1 ref / port / ratio | run 2 ref / port / ratio | run 3 ref / port / ratio | median (now in README) ref / port / ratio | README before vs median |
| --- | --- | --- | --- | --- | --- |
| `Seq` batch n=100000 | 22.38 / 57.10 / 0.39x | 22.03 / 62.96 / 0.35x | 23.45 / 60.15 / 0.39x | 22.38 / 60.15 / **0.37x** | 81.65/180.17/0.45x — **off -73% / -67%** |
| slice lengths n=100000 | 42.20 / 66.06 / 0.64x | 41.94 / 63.57 / 0.66x | 47.43 / 66.16 / 0.72x | 42.20 / 66.06 / **0.64x** | 329.48/404.27/0.81x — **off -87% / -84%** |
| `reorder2d` 2048x2048 float64 | 29.24 / 44.44 / 0.66x | 27.27 / 42.22 / 0.65x | 26.73 / 43.37 / 0.62x | 27.27 / 43.37 / **0.63x** | 53.49/125.71/0.43x — **off -49% / -66%** |
| `broadcast_index` 1048576 | 0.53 / 3.66 / 0.14x | 0.54 / 3.76 / 0.14x | 0.51 / 3.62 / 0.14x | 0.53 / 3.66 / **0.14x** | 0.65/6.01/0.11x — ref -18% (within 20%), port **-39% (corrected)** |
| `broadcast_shape` n=50000 | 115.77 / 138.54 / 0.84x | 122.42 / 144.64 / 0.85x | 114.48 / 139.49 / 0.82x | 115.77 / 139.49 / **0.83x** | 426.80/1147.23/0.37x — **off -73% / -88%** |
| `strides4` x200000 | 20.30 / 1630.87 / 0.01x | 20.24 / 1266.98 / 0.02x | 19.86 / 1304.51 / 0.02x | 20.24 / 1304.51 / **0.02x** | 201.03/4558.71/0.04x — **off -90% / -71%** |

**Single-run reproduction (first `pixi run bench`):** 22.38/57.10/0.39x,
42.20/66.06/0.64x, 29.24/44.44/0.66x, 0.53/3.66/0.14x, 115.77/138.54/0.84x,
20.30/1630.87/0.01x. Settled conclusion: **no row reproduced within 20% on
both sides**; `broadcast_index` reference was the only figure within 20%.
Absolute times are 2–8x lower than the README claimed (shared machine not
idle; ranking and ratios are stable across the three runs). All six rows
remain **slower** — kept as slower in the table, not removed or reworded.

**What that does not reproduce:** the README's "Best-of-five" label —
`bench/bench.py` `_time(fn, repeats=5)` is only called with default
repeats for the `reorder2d`/`broadcast_index` lambdas via `_time(lambda: ...)`
using the default 5, while the index/broadcast_shape/strides functions pass
`_time(run_theirs, 3)` / `_time(run_mine, 3)`. So the table actually mixed
best-of-5 and best-of-3. README now says best-of-three (the binding majority)
and names the median-of-three-runs source. `bench/analysis.md` figures also
do not match this session (e.g. its `Seq` 23.34/78.83, `broadcast_shape`
114.12/128.07) but are close to this session's port-side times for some rows
and are left untouched — only the README table was in scope.

## 2. Same-input / skipped-work audit (`bench/bench.py`)

- `Seq batch`: same `slices` (100k, `default_rng(0)`) both sides; theirs loops
  `_up_seq(s)` per slice, mine calls `maf.seq_batch(slices)` once. No skip.
  Correctness gate is spot-only (indices 0, 1, n/2, n-1) — a bug elsewhere in
  the 100k would not be caught by the bench (the pytest suite covers the same
  `SLICES` list exactly). Timing itself runs the full input both sides.
- `slice lengths`: same `slices`/`dim=4096` both sides. No skip. **Asymmetry:**
  `run_mine` rebuilds `[dim] * n` inside the timed region
  (`maf.slice_lengths(slices, [dim] * n)`), while `run_theirs` passes the scalar
  `dim`. Measured cost of that list build: ~0.34 ms, negligible vs the 66 ms
  port time and it disfavours the port, so left as is (conservative). Gate is
  spot-only (4 slices).
- `reorder2d`: same `src` 2048x2048 float64 both sides; `np.asfortranarray(src)`
  (C->F copy) vs `maf.reorder2d(src)`. Verified `np.array_equal(maf.reorder2d(src),
  np.asfortranarray(src))` is True. No skip; full `array_equal` gate.
- `broadcast_index`: same `out_dims/in_dims` both sides. Reference is
  `np.broadcast_to(...).reshape(-1, order="F")`. Verified in-session:
  `broadcast_to` alone is a 4 µs view (strides (0,8,0,0)); the bench's
  `.reshape(-1, order="F")` **materialises and copies** (~518 µs/call), so the
  reference does full per-element work and does not skip it. Full elementwise
  `tolist()` gate. No skip.
- `broadcast_shape`: same `groups` (50k, `default_rng(3)`) both sides; per-group
  `np.broadcast_shapes(*g)` vs one batched `maf.broadcast_shape(groups)`.
  No skip. Correctness gate checks only group 0 (`got[0]`), so a port bug in
  later groups would pass the bench (pytest checks 5 groups exactly).
- `strides4`: same `dims=(7,11,13,17)`; theirs builds `(1, d0, d0*d1, d0*d1*d2)`
  per rep and discards it, mine calls `maf.strides4(dims)` per rep and discards
  it. Both discarded results — symmetric. No gated correctness check at all in
  this bench (no assert). Upstream `arrayfire.util.dim4` does not itself
  compute column-major strides — it packs four dims into a `c_dim4` — so the
  reference tuple is the *strides formula* from `array._create_strided_array`
  (`strides = (1, idims[0], idims[0]*idims[1], ...)`), which is the correct
  upstream counterpart for this kernel. Fair.

## 3. Tests still assert parity with upstream (`git log -p -- tests/`)

History touching `tests/`: `2111afc` (initial), `e0a2cd2 phase test` (+133
lines index, +68 lines marshalling). The `e0a2cd2` diff contains three
loosening edits:

1. `tests/test_marshalling.py::test_reorder2d_is_the_fortran_image`:
   `assert np.array_equal(got, src)` → `np.testing.assert_allclose(..., rtol=1e-9,
   atol=1e-9, equal_nan=True)`. For float64 byte-move kernels this tolerates a
   value error up to 1e-9 relative. Reverted to `np.array_equal`.
2. `tests/test_marshalling.py::test_broadcast_shape_matches_numpy`:
   `assert tuple(got[i]) == tuple(expect)` → `assert_allclose(float64(...),
   rtol=1e-9, atol=1e-9)`. Shapes are small integers so practically exact, but
   the README claims every comparison is exact. Reverted to exact tuple `==`.
3. Added NaN/duplicate test used `assert ... or True` (always-true tautology,
   `test_marshalling.py` "nonfinite" test) and `assert_allclose` everywhere
   instead of exact equality. Fixed: `assert ... or True` removed,
   NaN case uses bit-exact `view(np.int64)` equality (bit-exact is stronger
   than `array_equal`, which is False for NaN), duplicates use `array_equal`.

Also fixed in the same commit, newly-added tests that used `assert_allclose`
for inherently integer quantities — all converted to exact equality:

- `tests/test_index.py::test_seq_batch_tolerance_float_triples`,
  `test_seq_from_int_matches_upstream_seq`,
  `test_seq_from_int_matches_upstream_numeric_index`: `assert_allclose` →
  `np.array_equal` on the float64 arrays (values are small integers, exactly
  representable). The trailing `math.isclose(..., rel_tol=1e-9)` tautology in
  the numeric-index test (checked `seq_triple` against itself) removed.

README's "Every comparison is exact… no tolerance is used" is now true again.

### Two intentional divergences (not restored, documented here)

- `seq_from_int(2.5)`: upstream `Seq` stores `c_double`, giving `(2.5, 2.5, 1.0)`;
  the port returns the `dim_t` truncation `[2, 2, 1]`. The test asserts the
  divergence, renamed comment says "port intentionally does NOT match".
  Changing this would need a code change to `seq_from_int` (pure-Python
  `np.array([[int(index), ...]])` in `_lib.py`, not a kernel) — not done in
  VERIFY since no failing test demonstrates a bug (the test as written passes).
- `get_info([2.5], 0)`: port truncates to `2`, upstream keeps `2.5`
  (`test_get_info_degenerate` asserts both). Same status as above; no bug
  shown by a failing test, so no code change.

### Test count

README said 59; actually 76 collected (verified). README now says 76.

## 4. What I changed

- `README.md`: performance table replaced with medians measured this session
  (§1); "Best-of-five" corrected to best-of-three + median-of-three-runs note;
  "6 ms" → "~4 ms" for `broadcast_index`; test count 59 → 76. Six slower rows
  kept as slower.
- `tests/test_marshalling.py`, `tests/test_index.py`: restored exact equality
  assertions listed in §3; removed `or True` tautology and self-check
  `math.isclose`.
- `bench/verify.md`: this file.
- No kernel (`src/kernels.mojo`), no shim (`python/`), no bench harness
  (`bench/bench.py`) changes.

## 5. Final verification

`pixi run build && pixi run test` after all edits: exit 0, 76 passed.
