# mojo-arrayfire

`mojo-arrayfire` is the Python-side arithmetic of
[ArrayFire](https://www.arrayfire.org/), with the loops written in Mojo. The
Python package is named `mojo_arrayfire`, so it installs alongside the real
`arrayfire` rather than shadowing it, and it never imports it.

```python
import mojo_arrayfire as maf

maf.seq_batch([slice(2, 5), slice(None, None, -1)])
# array([[ 2,  4,  1],
#        [-1,  0, -1]])
maf.slice_length(slice(0, 6, 2), dim=10)          # 3
maf.assign_shape((slice(0, 5),), (10, 20, 30, 40)) # array([ 5, 20, 30, 40])
maf.broadcast_index((2, 3, 1, 1), (1, 3, 1, 1))   # array([0, 0, 1, 1, 2, 2])
```

## What is ported, and why

The `arrayfire` Python package is a ctypes binding. Almost every entry point in
its 12,700 lines is `safe_call(backend.get().af_xxx(...))`, and the actual
arithmetic lives in the C library, which is not installed here
(`import arrayfire` fails with "Could not load any ArrayFire libraries"). What
is left in Python, and what genuinely executes on every call, is the metadata
and index arithmetic wrapped around those calls, plus the host-side data
marshalling. That is what this port covers.

| area | implemented API | kernel | upstream function |
| --- | --- | --- | --- |
| Slice to ArrayFire `Seq` | `seq_batch`, `seq_from_int` | `af_seq_batch` | `arrayfire.index.Seq.__init__` |
| Slice element count | `slice_length`, `slice_lengths` | `af_slice_lengths` | `arrayfire.array._slice_to_length` |
| Assignment shape | `assign_shape`, `assign_dims` | `af_assign_dims` | `arrayfire.array._get_assign_dims` |
| Row-major to column-major host buffer | `reorder2d` | `af_reorder2d` | `arrayfire.interop._cc_to_af_array` plus `Array._reorder` |
| Broadcast operand shapes | `broadcast_shape` | `af_broadcast_shape` | `af.broadcast` shape rule |
| Broadcast gather map | `broadcast_index` | `af_broadcast_index` | the index map behind `af.broadcast` |
| Column-major strides | `strides4` | `af_strides4` | `arrayfire.util.dim4` |
| Dtype promotion of a scalar | `implicit_dtype` | stays in Python | `arrayfire.util.implicit_dtype` |
| Shape and element count | `get_info` | stays in Python | `arrayfire.array._get_info` |

`implicit_dtype` and `get_info` are listed as not ported on purpose: one is a
handful of integer comparisons on a scalar and the other is four multiplications
on a four-element shape. Neither has a loop worth compiling, so shipping a
kernel for either would be a fake kernel. They are implemented in Python, and
tested against the real upstream source.

The batching is the substantive change. Upstream builds one `Seq` per index
axis, in Python, per `__getitem__`; this builds the whole four-axis index record
in one compiled pass.

## Not implemented

- Every device call. `af_add`, `af_matmul`, `af_convolve`, `af_sort`,
  `af_fft` and the rest live in the ArrayFire C library, not in this package, and
  are left to the real `arrayfire`.
- `af.Array` itself: allocation, lifetime, `dtype()`, `strides()`, `to_ctype()`.
  Those are handle bookkeeping around the C library.
- `af.Array` and `af.ParallelRange` keys in `_get_assign_dims`. Resolving a
  boolean mask's nonzero count needs a live array handle; the number and slice
  branches are ported and the array branches raise.
- Keyed indexing with a device-side index array.
- `arrayfire.array._ctype_to_lists`, and therefore `Array.to_list`. Its index
  map is the identity for every layout ArrayFire produces, so there is no
  arithmetic to compile: a column-major flat buffer sliced into blocks of
  `shape[0]` is the same as reshaping it. The function is also wrong for three
  or more axes, where it advances the offset by `shape[0]` at every level and
  re-reads the lowest block. That is documented here rather than ported.
- Anything from `image.py`, `signal.py`, `ml.py`, `sparse.py`, `lapack.py`,
  `blas.py`, `graphics.py` beyond the metadata arithmetic above; all of it is a
  thin `safe_call` wrapper.

## Parity, and how it was checked

`import arrayfire` cannot run in this environment, so the parity oracle is the
upstream Python source itself. `tests/_upstream.py` reads the installed
distribution from the test virtualenv, extracts the relevant class and function
source with a regex, and executes it verbatim, supplying only the names those
definitions close over: `ctypes`, `numbers`, `math`, and the `Dtype` enum
rebuilt from `library.py`. The oracle is therefore upstream's code, not a
paraphrase of it. If the source cannot be found, those tests skip loudly rather
than falling back to a local copy.

Every comparison is exact. These are integer indices, shape arithmetic and
byte moves; there is no floating-point arithmetic in any kernel, so there is
nothing for FMA to perturb and no tolerance is used or needed.

One subtlety is worth naming. Upstream's slice length is
`int((stop - start - 1) / step + 1)`, where the `+ 1` is *inside* the
truncation. Truncating the quotient and then adding one gives a different
answer on a negative fractional quotient, so the kernel folds the `+ step` in
before dividing. `test_slice_length_matches_upstream` covers the
negative-step cases that catch this.

## Install and build

```bash
pixi install
pixi run build      # -> dist/libmojo-arrayfire.so
pixi run test
pixi run bench
```

In this shared environment, source `/nvme0n1-disk/mojo-toolchain/activate.sh`
first and then:

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

## Performance

Best-of-five wall clock, same process, every case gated on an exact correctness
check before timing. The reference column is the real upstream code for the
index cases, executed one call per element, and the fastest reasonable NumPy
formulation for the rest. The machine is shared and was not idle, so treat
absolute times as indicative.

| case | reference | mojo-arrayfire | result |
| --- | ---: | ---: | ---: |
| `Seq` batch n=100000 | 81.65 ms | 180.17 ms | **0.45x, slower** |
| slice lengths n=100000 | 329.48 ms | 404.27 ms | **0.81x, slower** |
| `reorder2d` 2048x2048 float64 | 53.49 ms | 125.71 ms | **0.43x, slower** |
| `broadcast_index` 1048576 elements | 0.65 ms | 6.01 ms | **0.11x, slower** |
| `broadcast_shape` n=50000 | 426.80 ms | 1147.23 ms | **0.37x, slower** |
| `strides4` x200000 | 201.03 ms | 4558.71 ms | **0.04x, slower** |

Every case is slower, and the reasons are structural rather than fixable by
tuning a kernel:

- The `Seq`, slice-length and broadcast-shape cases are dominated by the shim's
  Python encoding, not by the kernel. Turning 100,000 Python `slice` objects
  into three int64 arrays costs more than the arithmetic the kernel then does
  in one call. Batching amortises the FFI, and that is all it amortises; the
  encoding is irreducible while the public API takes Python objects.
- `strides4` is four multiplications. The reference builds the same tuple in
  Python; the port pays ctypes overhead to do the same work. The port should
  not exist, and the README says so.
- `broadcast_index` is the one case with real per-element work, and NumPy's
  equivalent is a single vectorised `broadcast_to`. One million elements in
  6 ms is a compiled index map; the vectorised form does the same job with no
  per-element branch at all.
- `reorder2d` is a byte-granular permutation. NumPy's `asfortranarray` uses a
  blocked, type-aware copy; the byte loop is a straight per-element move. A
  typed, blocked kernel would close some of this, and is the obvious next step
  if the port is ever used on a hot path.

None of these kernels is threaded. They are memory bound and short, and the
toolchain's own guidance is that threading a bandwidth-bound loop makes it
slower.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-arrayfire.so`.

`python/mojo_arrayfire` owns every array: it validates, encodes and allocates,
then makes one call per batch. Buffers cross the C ABI as 64-bit addresses and
are reconstructed in Mojo as `Pointer[Int64, AnyOrigin[mut=True]]`, which keeps
the exported symbols non-parametric. No allocation happens inside Mojo, and the
odometer state lives in locals rather than a caller-provided scratch buffer.

Two implementation notes worth carrying to the next port. A read-modify-write
through a pointer element (`p[unsafe_offset=i] += 1`) does not survive this
toolchain's optimiser when the pointer is to caller memory, so odometers are
kept in locals. And every array handed to a kernel is bound to a local before
its address is taken: an array built inline in the call expression has no
reference left once `_addr` has returned, and its buffer can be recycled before
the kernel reads it.

## Tests

`PYTHONPATH=python python -m pytest tests -q` — 59 tests.

They are chosen so that a plausible kernel bug fails them: a missing negative
step swap in the `Seq` mapping (`test_seq_batch_pinned_values`), a truncation
before instead of after the `+ 1` (`test_slice_length_matches_upstream` on
negative-step slices), an axis-padding mistake in the assignment shape
(`test_assign_dims_pads_missing_axes_with_a_full_slice`), source strides taken
from the output shape instead of the operand shape
(`test_broadcast_index_gathers_like_numpy`), a zero stride missing on a
broadcast axis (`test_broadcast_index_repeated_axis_gets_a_zero_stride`), a
transposed element move (`test_reorder2d_matches_the_cc_to_af_map`), and a
`base` that forgot the per-operand stride in a batch
(`test_broadcast_shape_batch_is_elementwise`).

## License

3-clause BSD, matching ArrayFire.
