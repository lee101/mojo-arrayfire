# mojo-arrayfire — measured hotspot analysis (ANALYZE phase)

Measured map only; no algorithm changed. Every number below comes from a run in
this session (flock-wrapped so no other benchmark overlapped). `src/` and
`tests/` untouched; this file is the sole addition.

## 0. Provenance

- `pixi run bench` (i.e. `flock /tmp/mojo-bench.lock python bench/bench.py`), best-of-3, exit 0:

  | case | reference | mojo-arrayfire | ratio |
  | --- | ---: | ---: | ---: |
  | `Seq batch n=100000` | 23.34 ms | 78.83 ms | **0.30x, slower** |
  | `slice lengths n=100000` | 56.32 ms | 81.36 ms | **0.69x, slower** |
  | `reorder2d 2048x2048 float64` | 27.83 ms | 38.85 ms | **0.72x, slower** |
  | `broadcast_index 1048576` | 0.49 ms | 3.46 ms | **0.14x, slower** |
  | `broadcast_shape n=50000` | 114.12 ms | 128.07 ms | **0.89x, slower** |
  | `strides4 x200000` | 18.15 ms | 1813.78 ms | **0.01x, slower** |

- Upstream `arrayfire` C library is **not importable** in this environment
  (`import arrayfire` → `MaxRayParallelism` env check passes but the C backend
  shared library is absent, same as README says). The reference column is
  therefore the upstream Python source executed verbatim via `tests/_upstream.py`
  (one call per element, matching how the binding builds an index record) for the
  index cases, and the fastest reasonable NumPy formulation for the rest.
  I re-verified the oracle inputs in-session: `_upstream.seq_triple(slice(2,5)) ==
  (2.0, 4.0, 1.0)`, `_get_assign_dims((slice(0,5),), [10,20,30,40]) == [5,20,30,40]`,
  `_get_info((3,4),0) == (2, [3,4,1,1])`, `implicit_dtype(1.5, f32) == 0`.
- Split harnesses were run in-session via `pixi run python` and removed
  after measurement to leave only `bench/analysis.md` per the rules.  The
  bench/README absolute times differ run-to-run on this shared 72-CPU box;
  rank order and glue/kernel split are stable across runs in this session.

## 1. Ranked hotspot table

Rank = share of end-to-end time in the wrapper as called by real callers at the
sizes below. `bytes touched` = read + write of the *kernel* buffers only
(Python object graph excluded — see column notes). Flops are integer ops
(add/mul/div/compare) counted once per element/axis from the kernel source;
`flops/byte` is flops ÷ bytes touched. All six port rows are slower than the
reference — marked **SLOWER**.

| rank | function | input size | time (measured this session) | bytes touched (kernel) | flops | flops/byte | bound | time sits in |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `strides4` (`af_strides4`) | 200,000 calls, 4-elem dims | **1,814 ms full (bench); 5.3–6.1 µs/call single-call** | 64 B/call (32 read + 32 write) | ~7/call (1 init + 3 mul + 4 store) | ~0.11 | memory (trivial) | **Python/ctypes glue ~100%**; kernel ≈ 0 |
| 2 | `seq_batch` (`af_seq_batch`) | n=100,000 slices | **68–79 ms full; 0.66 ms kernel-only; 58–64 ms encode-only** | 4.8 MB (starts+stops+steps 2.4 MB read, out 2.4 MB write) | ~1.0 M (~10/slice: compares, branches, sub/add) | ~0.21 | memory | **Python encoding ~94%**, kernel ~1% |
| 3 | `slice_lengths` (`af_slice_lengths`) | n=100,000 slices + dims | **82 ms full; 1.27 ms kernel-only; 57 ms encode-only; +4.5 ms `_i64(dims)`** | 4.0 MB (dims+starts+stops+steps 3.2 MB read, out 0.8 MB write) | ~0.6 M (~6/slice incl. 1 div via `_tdiv`) | ~0.15 | memory | **Python encoding ~70%**, kernel ~1.6% |
| 4 | `broadcast_shape` (`af_broadcast_shape`) | n=50,000 groups, 2 operands × 4 axes | **128–191 ms full; 1.07–1.13 ms kernel-only; 119 ms canonicalise + 39 ms flatten** | 2.4 MB (shapes 1.6 MB read, out 0.8 MB write) | ~1.6 M (~32/group: 4 axes × 2 ops × ~4) | ~0.67 | memory | **Python canonicalisation ~65–95%**, kernel ~0.6% |
| 5 | `reorder2d` (`af_reorder2d`) | 2048×2048 float64 = 4,194,304 elems | **39–43 ms full; 33–34 ms kernel-only; 28 ms NumPy** | 67.1 MB (33.6 MB read + 33.6 MB write) | ~0 (pure data move; ~8.4 M index mul/add in nested loops) | 0.00 | memory | **kernel ~78%** (≈2.0 GB/s r+w), glue ~22% |
| 6 | `broadcast_index` (`af_broadcast_index`) | 1,048,576 out elements | **3.3–3.5 ms full; 3.17–3.47 ms kernel-only; 0.51 ms NumPy** | 8.4 MB (out write only; dims negligible) | ~5.2 M (4 mul + 3 add + 2–3 compare per element) | ~0.62 | memory | **kernel ~96%**, glue ~4% |

Additional measured rows (small / non-loop paths, same session):

| function | input size | time | kernel? | verdict note |
| --- | --- | --- | --- | --- |
| `slice_length` (single, via batch-of-1) | 1 slice | 13.0 µs/call | yes (batched, n=1) | ctypes + encode dominate; use `slice_lengths` for batches |
| `assign_shape` / `assign_dims` (n=4 fixed) | 1 key, ≤4 axes | 16.2 µs/call | yes (n=4) | glue-bound; real callers pass one key per `__setitem__` |
| `seq_from_int` | 1 int | 0.98 µs/call | **no** (pure Python `np.array`) | already no kernel; nothing to do |
| `implicit_dtype` | 1 scalar | 1.05 µs/call | **no** (pure Python, by design) | already no kernel; nothing to do |
| `get_info` | 1 short shape | 0.95 µs/call | **no** (pure Python, by design) | already no kernel; nothing to do |
| `broadcast_index` tiny | 6 elems | 10.1 µs/call | yes | per-call ctypes floor ~10 µs |
| `seq_batch` tiny | n=1 / n=1,000 | 10.0 µs / 346 µs (346 ns/elem) | yes | encode dominates even at n=1k |

`split during this session` — why the table `time` for ranks 2–4 is far above
the kernel-only figure, and ranks 5–6 is not.

## 2. Where the time actually sits — per-function evidence

### 2.1 `strides4` — pure shipping overhead (rank 1)

- Single call best-of-5000: **5.3–6.1 µs**. 200,000 calls × 5.5 µs ≈ 1.1–1.8 s,
  matching the bench 1,814 ms. The kernel is 4 stores + 3 multiplies.
- `python/mojo_arrayfire/_lib.py`: `tuple(int(x) for x in dims)` + `_i64(dims)`
  (one `np.ascontiguousarray`) + `np.empty(4)` + one ctypes call per invocation.
- There is **no Python loop re-entering ctypes per element** — it is worse: one
  ctypes call *per logical call* for a 4-wide multiply. `assign_shape` (16 µs)
  and `slice_length`-single (13 µs) show the same floor. Upstream's `dim4` is a
  Python tuple.
- Verdict: **none** (see §4). The port should not exist on a hot path — README
  says the same. If kept, it is `none`, not `simd`: there is nothing to vectorise,
  just shipping.

### 2.2 `seq_batch` — Python encoding of 100k `slice` objects (rank 2)

- Full: 68–79 ms. Kernel-only (`lib.af_seq_batch` on pre-encoded int64 buffers):
  **0.66 ms** (6–7 ns/element, scales linearly to n=1M: 6.55 ms). Encode-only
  (`_triples` → `[_encode_slice(s) for s in slices]` + 3× `_i64`): **58–64 ms**.
  `np.empty(3n)`: ~1 µs. So **~94% Python, ~1% kernel**.
- The re-entry: `_triples` builds `enc = [_encode_slice(s) for s in slices]`,
  a list of 100,000 3-tuples, then three list comprehensions `[e[0] for e in
  enc]` etc., then three `np.ascontiguousarray` calls. A plain-Python triple
  loop doing the same field extraction (no numpy) is 21.9 ms — the three-pass
  + `_i64` packaging costs ~3× that. A single transposed loop appending to
  three lists measured 49.7 ms (vs 58–64 ms).
- Upstream reference is 23 ms (one `Seq.__init__` per slice in Python —
  `tests/_upstream.py` executes it verbatim). The port loses because it *adds*
  a build-three-int64-arrays encoding on top of the same per-slice Python work,
  to save a kernel that was only 0.7 ms to begin with.

### 2.3 `slice_lengths` — same encoding, plus a dims copy (rank 3)

- Full: 82 ms. Kernel-only: **1.27 ms**. Encode (`_triples`): **57 ms**.
  `_i64(dims)` where `dims = [dim]*100000`: 4.5 ms, plus 0.28 ms to build
  `[dim]*n` in the bench lambda itself. Shares of the remainder: encode ~70%,
  kernel ~1.6%.
- Upstream reference this session: 44.9 ms (verbatim `_slice_to_length` per
  slice). The kernel divides by the step via `_tdiv` (truncation-toward-zero)
  — integer division is the only non-trivial op, once per slice.
- Per-call allocations: `starts/stops/steps` (3× 0.8 MB), `d` (0.8 MB),
  `out` (0.8 MB) — five 100k-int64 buffers per call, all built in Python
  before one ctypes call. All are touched once by the kernel.

### 2.4 `broadcast_shape` — `int()`-per-element canonicalisation loop (rank 4)

- Full: 128–191 ms (run variance on shared box; bench 128 ms). Kernel-only:
  **1.07–1.13 ms**. Canonicalisation `groups = [tuple(tuple(int(x) for x in
  op) for op in g) for g in groups]`: **119 ms**. Flatten `[v for g in groups
  for op in g for v in op]` + `_i64`: **39 ms**. Bad-group `(result<0).any`
  + `argmax` path: ~0.9 ms.
- The canonicalisation is a **Python-level nested loop that re-enters per
  element**: `int(x)` is called 400,000 times (50k groups × 2 ops × 4 axes).
  Only then does one `af_broadcast_shape` ctypes call run.
- Upstream reference this session: 114 ms (`np.broadcast_shapes` per group).
  The port is at parity-to-slightly-slower because its Python canonicalisation
  (≈160 ms) is heavier than upstream's per-group NumPy call (≈114 ms), and the
  kernel saving (1 ms) cannot pay for it.

### 2.5 `reorder2d` — a byte loop; the one honest kernel hotspot (rank 5)

- Full: 39–43 ms. Kernel-only: **33–34 ms**. NumPy `asfortranarray`: **28 ms**.
  `np.ascontiguousarray(src)` no-ops when `src` is already C-contiguous (0.0 ms
  measured), `reshape(-1)` is a 0.4 µs view. Glue ≈ 22% of the gap; kernel ≈
  78% and is the real cost.
- The kernel (`src/kernels.mojo §af_reorder2d`) is a **triple nested loop over
  bytes**: `for i1: for i0: for b in range(esize): dst[dbase+b] = src[sbase+b]`
  with `esize = 8` for float64. That is ~33.5 M byte loads/stores plus ~8.4 M
  index multiply-adds. Effective throughput **~2.0 GB/s read+write** (measured
  2048×2048), falling to **0.48 GB/s at 2048×8192** — non-linear collapse at
  large sizes (cache/TLB: strided `i0*n1` access on the read side). At
  512×512 it is 2.26 GB/s and 1024×1024 2.35 GB/s; the 2048-square already
  degrades.
- NumPy wins not by data movement but by a **blocked, type-aware copy** —
  one 8-byte move per element with cache blocking vs one byte per iteration.

### 2.6 `broadcast_index` — odometer loop, the other honest kernel (rank 6)

- Full: 3.3–3.5 ms. Kernel-only: **3.17–3.47 ms** (96%). NumPy `broadcast_to`:
  **0.51 ms**. `np.empty(1M)` is 5 µs; `_i64` on the two 4-elem dim tuples is
  1.7 µs. Glue is negligible.
- The kernel is an odometer: 4 multiplies + 3 adds + 2–3 comparisons per
  output element (`out[t] = i0*t0+i1*t1+i2*t2+i3*t3; i0+=1; carry...`).
  Write throughput ~2.4 GB/s (8 bytes × 1M / 3.3 ms), scaling linearly
  (0.35 ms @ 65k, 13.5 ms @ 4.2M). NumPy's `broadcast_to` materialises **no
  per-element index array at all** — it returns a strided view with stride 0
  on the broadcast axis. The port pays full materialisation to produce a map
  whose only consumer is a gather `src[indices]`; where it is consumed as a
  gather, fusing the gather (or using the view) removes the pass entirely.

## 3. Flags — loops and allocations the next phases must not ignore

1. **Python-level loops that re-enter per element** (no ctypes per element
   — the batching *did* amortise that — but Python per element):
   - `seq_batch` / `slice_lengths`: `[_encode_slice(s) for s in slices]` then
     three `[e[k] for e in enc]` passes → 4 Python passes over n tuples +
     3 numpy packs. `broadcast_shape`: `tuple(tuple(int(x) …))` per group
     (400k `int()` calls) + triple-nested flatten list-comp.
   - `broadcast_shape` canonicalisation is the single largest pure-Python block
     (119 ms). Every element is visited 3× in Python before the kernel sees it
     once.
2. **Per-call allocations / copies** (all before a single ctypes call):
   - `seq_batch`: 3× `n`-int64 encode buffers + 1× `3n` output.
   - `slice_lengths`: 4× `n`-int64 (`dims` + 3 encode) + 1× `n` output; the
     caller also builds `dims = [dim]*n` first.
   - `broadcast_shape`: canonicalised nested tuples (object graph), flattened
     `nops*4*n` int64 buffer, `4n` output.
   - `reorder2d`: `np.ascontiguousarray(src)` copies when input is
     non-contiguous (no-op measured here), plus a fresh `src.size`-element
     output. `reshape(-1)` is a view.
   - `broadcast_index`: `np.empty(total)` int64 — the output *is* the cost
     (8.4 MB for 1M).
3. **Per-call ctypes floor**: ~5.5 µs (`strides4`), ~10 µs (`broadcast_index`
   tiny, `seq_batch` n=1), 13 µs (`slice_length` single), 16 µs (`assign_shape`).
   Any API used once per `__getitem__`/`__setitem__` (single slice, single key)
   lives on this floor. `assign_shape`/`slice_length`-single/`strides4` should
   never be called per-element by fixed callers.
4. **No kernel is threaded, vectorised, or GPU-backed** (`src/kernels.mojo`:
   scalar `for` loops only; byte-loop in `af_reorder2d`; odometer in
   `af_broadcast_index`). `nproc` = 72 — parallelism has headroom, but see §4
   for why most rows must not use it.

## 4. Next-phase verdicts (evidence-backed)

Rules applied: GPU needs `flops/byte` ≳ 2 and enough per-call work to amortise
copy-in/out; SIMD needs a wide, contiguous, regular inner loop; parallel needs
large n with no reuse; `none` = remove the call or keep in Python.

| rank | function | flops/byte | next phase | evidence |
| --- | --- | --- | --- | --- |
| 1 | `strides4` | ~0.11 | **none** | 7 integer ops/call; ctypes floor (5.5 µs) is 100% of cost. No unroll/simd/parallel/gpu can beat a Python tuple. Do not call the kernel on hot paths; keep Python or inline. |
| 2 | `seq_batch` | ~0.21 | **none** (for the kernel); *unroll does not apply* — the cost is the Python wrapper, not the kernel | Kernel is 0.66 ms at 0.21 f/b; even a 10× kernel win saves 0.6 ms of 70 ms. Vectorising the encoder is impossible (input is Python `slice` objects). The real fix is a different ingestion path (pre-encoded arrays / accept triples directly), which is an API change, not unroll/simd/parallel/gpu. GPU at 0.21 f/b is disqualified — needs ≳2. |
| 3 | `slice_lengths` | ~0.15 | **none** (same as seq) | Same: 1.27 ms kernel vs 57 ms encode. Integer division is serial-ish and the batch is already flat. Parallel on the kernel would save ≤1 ms. |
| 4 | `broadcast_shape` | ~0.67 | **none** (kernel); Python canonicalise must go before any kernel phase matters | Kernel 1.1 ms vs ~160 ms Python. 0.67 f/b is the highest of the glue-bound rows but GPU still disqualified (≪2 f/b, and per-group data is 32 B — launch overhead > work). |
| 5 | `reorder2d` | 0.00 | **simd** (typed + blocked), secondarily **parallel** if large | The only row where the kernel dominates *and* is fixable in place. 0 f/b, 2.0 GB/s (0.48 GB/s at 2048×8192) proves bandwidth-bound, cache-limited, byte-at-a-time movement. Typed 8-byte (f64) or 4-byte moves + blocking + cache-tile the transpose; `simd_width_of` load/store is available. Parallel (`parallelize` with explicit worker count) is justified above ~1M elements (72 cores, independent tiles) once blocking exists; without blocking it will contend on the same cache lines. GPU disqualified: 0 f/b ≪ 2. |
| 6 | `broadcast_index` | ~0.62 | **simd** (odometer → strided/blocked form) or **none** if the consumer can take a `broadcast_to` view instead | 5.2 M integer ops over 8.4 MB written; currently scalar odometer with branches per element. 0.62 f/b disqualifies GPU (≪2). SIMD is the bet: reformulate as block-strided arithmetic (per-axis counters vectorised, or nested block loops so the carries are rare). Parallel is second: the loop is embarrassingly parallel by output tile and scales linearly, so it can help at 4 M+ elements, but SIMD+blocking captures the same win at 1 M without thread launch cost. |

### Explicit non-verdicts (tested, rejected on evidence)

- **GPU for any row**: no row clears judicial review. Highest `flops/byte` is
  0.67 (`broadcast_shape`); every other row is ≤0.62, most <0.25, two are ~0.
  The 2-flop/byte bar for a GPU path to pay its host round-trip is not met
  anywhere, and per-call payloads (32 B to 8 MB) are launch-dominated. This is
  the toolchain's `~2 flops per byte` rule applied to measured figures, not a
  guess.
- **Unroll for any row**: unrolling a scalar integer branch (`af_seq_batch`,
  `af_slice_lengths`) saves cycles inside a 1%-of-wall-clock kernel. Measured
  kernel share is 1.0%/1.6%; there is nothing to unroll that moves the needle.
- **Parallel for glue-bound rows (ranks 1–4)**: threading a bandwidth-bound
  Python encoding loop (ranks 2–4) or a 7-op ctypes call (rank 1) makes it
  slower, not faster — the bottleneck is Python object traversal and per-add
  `int()` conversion, which threads cannot shorten.

## 5. Bottom line for phase planners

- **6 of 6 ported hot entry points are slower than reference at bench sizes**
  (ratios 0.01–0.89x). None should ship as-is on a hot path without a phase
  below acting first.
- Three rows (2–4) are **Python-encoders with a kernel attached** — the
  encoder is 65–94% of their time. Any future phase that polishes the kernel
  without removing/re-encoding that wrapper is polishing 1% of the line.
- Two rows (5–6) are **honest kernels**: `reorder2d` (byte loop, 2 GB/s,
  blocking+typed SIMD is the lever) and `broadcast_index` (odometer, SIMD or
  fuse/avoid-materialisation is the lever). `reorder2d` additionally degrades
  superlinearly at 2048×8192 (0.48 GB/s) — cache blocking is not optional
  there.
- One row (1) is a **non-kernel**: remove per-call ctypes for a 4-wide
  multiply.

Reproduce: `pixi run bench` (flock-wrapped; per-function splits were run
in-session and are summarised in §2).
