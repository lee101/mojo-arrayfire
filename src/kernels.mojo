"""ArrayFire metadata, index and host-marshalling arithmetic in Mojo.

The `arrayfire` Python package is a ctypes binding: almost every entry point is
`safe_call(backend.get().af_xxx(...))`. What is left in Python, and what actually
executes on every call, is the metadata and index arithmetic wrapped around
those calls, plus the host-side data marshalling (`to_ndarray`, `to_list`).
That is what this unit ports.

Every exported symbol takes buffer addresses as plain `Int` and rebuilds the
pointer inside the body, because `@export` rejects parametric functions and an
inferred pointer origin would make the symbol parametric.

Sentinel for "argument absent": `af_slots` receives -2**62 for a missing
`slice.start` or `slice.stop` and 0 for a missing `slice.step`. A real step is
never 0, so one sentinel per slot is enough. `slice.py` rejects a zero step
outright.
"""

comptime I64P = Pointer[Int64, AnyOrigin[mut=True]]
comptime NONE = Int64(-4611686018427387904)  # -2**62


def _i64p(a: Int) -> I64P:
    return I64P(unsafe_from_address=a)




def _tdiv(a: Int64, b: Int64) -> Int64:
    """Division truncated toward zero, which is what Python's `int(x / y)` does.

    Worked on magnitudes so the floor-versus-truncate question never arises.
    """
    var neg = (a < 0) != (b < 0)
    var x = a
    if x < 0:
        x = -x
    var y = b
    if y < 0:
        y = -y
    var q = x // y
    if neg:
        q = -q
    return q


@export("af_seq_batch")
def af_seq_batch(
    n: Int, start_addr: Int, stop_addr: Int, step_addr: Int, out_addr: Int
) abi("C"):
    """Batch of Python slices -> ArrayFire `Seq(begin, end, step)` triples.

    Faithful to `arrayfire.index.Seq.__init__`, including the two empty-selection
    special cases and the trailing `end - copysign(1, step)` adjustment. The
    `isinstance(S, numbers.Number)` branch of upstream has no slice to encode
    and is handled by the shim.
    """
    var start = _i64p(start_addr)
    var stop = _i64p(stop_addr)
    var step_in = _i64p(step_addr)
    var out = _i64p(out_addr)
    for i in range(n):
        var begin = Int64(0)
        var end = Int64(-1)
        var step = Int64(1)
        var s = step_in[unsafe_offset=i]
        if s != 0:
            step = s
            if s < 0:
                begin = Int64(-1)
                end = Int64(0)
        var bs = start[unsafe_offset=i]
        var be = stop[unsafe_offset=i]
        if bs != NONE:
            begin = bs
        if be != NONE:
            end = be
        if begin >= 0 and end >= 0 and end <= begin and step >= 0:
            begin = Int64(1)
            end = Int64(1)
            step = Int64(1)
        elif begin < 0 and end < 0 and end >= begin and step <= 0:
            begin = Int64(-2)
            end = Int64(-2)
            step = Int64(-1)
        if be != NONE:
            if step > 0:
                end -= 1
            else:
                end += 1
        var base = i * 3
        out[unsafe_offset=base] = begin
        out[unsafe_offset=base + 1] = end
        out[unsafe_offset=base + 2] = step


@export("af_slice_lengths")
def af_slice_lengths(
    n: Int, dim_addr: Int, start_addr: Int, stop_addr: Int, step_addr: Int,
    out_addr: Int
) abi("C"):
    """Batch of `arrayfire.array._slice_to_length`: a slice's element count.

    Negative bounds are folded as `dim - bound`, which is upstream's rule. The
    count is upstream's `int((stop - start - 1) / step + 1)`, where the `+ 1`
    is inside the truncation: in exact arithmetic that is
    `int((stop - start - 1 + step) / step)`, which is what this computes.
    Truncating the quotient first and adding one afterwards would differ on a
    negative fractional quotient, so the `+ step` is folded in first.
    """
    var dims = _i64p(dim_addr)
    var start = _i64p(start_addr)
    var stop = _i64p(stop_addr)
    var step_in = _i64p(step_addr)
    var out = _i64p(out_addr)
    for i in range(n):
        var dim = dims[unsafe_offset=i]
        var t0 = start[unsafe_offset=i]
        var t1 = stop[unsafe_offset=i]
        var t2 = step_in[unsafe_offset=i]
        if t0 == NONE:
            t0 = 0
        elif t0 < 0:
            t0 = dim - t0
        if t1 == NONE:
            t1 = dim
        elif t1 < 0:
            t1 = dim - t1
        if t2 == 0:
            t2 = 1
        out[unsafe_offset=i] = _tdiv(t1 - t0 - 1 + t2, t2)


@export("af_assign_dims")
def af_assign_dims(
    n: Int, kind_addr: Int, dim_addr: Int, start_addr: Int, stop_addr: Int,
    step_addr: Int, out_addr: Int
) abi("C"):
    """Batch of `arrayfire.array._get_assign_dims` for number and slice keys.

    `kind` is 0 for a plain integer index (which pins the axis to 1) and 1 for a
    slice. Keys that address an `af.Array` need a live device handle and are
    resolved in the shim, which is where upstream handles them too.
    """
    var kinds = _i64p(kind_addr)
    var dims = _i64p(dim_addr)
    var start = _i64p(start_addr)
    var stop = _i64p(stop_addr)
    var step_in = _i64p(step_addr)
    var out = _i64p(out_addr)
    for i in range(n):
        var kind = kinds[unsafe_offset=i]
        var dim = dims[unsafe_offset=i]
        if kind == 0:
            out[unsafe_offset=i] = 1
        else:
            var t0 = start[unsafe_offset=i]
            var t1 = stop[unsafe_offset=i]
            var t2 = step_in[unsafe_offset=i]
            if t0 == NONE:
                t0 = 0
            elif t0 < 0:
                t0 = dim - t0
            if t1 == NONE:
                t1 = dim
            elif t1 < 0:
                t1 = dim - t1
            if t2 == 0:
                t2 = 1
            out[unsafe_offset=i] = _tdiv(t1 - t0 - 1 + t2, t2)


@export("af_reorder2d")
def af_reorder2d(
    n0: Int, n1: Int, esize: Int, src_addr: Int, dst_addr: Int
) abi("C"):
    """Row-major (n0, n1) host buffer -> column-major, the `_cc_to_af_array` map.

    `arrayfire.interop._cc_to_af_array` reverses the shape and then calls
    `af_reorder` with the axes swapped; on the host that is exactly
    `dst[i0 + i1*n0] = src[i1 + i0*n1]`. Same mapping, one pass, no device.

    The move is element-granular and the index map does not depend on the
    element type, so this walks bytes with `esize` bytes per element and serves
    every dtype from one kernel.
    """
    var src = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=src_addr)
    var dst = Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=dst_addr)
    for i1 in range(n1):
        for i0 in range(n0):
            var sbase = (i1 + i0 * n1) * esize
            var dbase = (i0 + i1 * n0) * esize
            for b in range(esize):
                dst[unsafe_offset=dbase + b] = src[unsafe_offset=sbase + b]


@export("af_broadcast_shape")
def af_broadcast_shape(
    n: Int, nops: Int, shapes_addr: Int, out_addr: Int
) abi("C"):
    """Elementwise broadcast of operand shapes, ArrayFire's rule.

    A dimension of extent 1 is broadcast against its partner; two dimensions
    above 1 must agree. A group that violates the rule is reported as -1 in
    every axis, which is how a shape error surfaces without raising from Mojo.
    """
    var shapes = _i64p(shapes_addr)
    var out = _i64p(out_addr)
    for g in range(n):
        var base = g * nops * 4
        for d in range(4):
            var dim = Int64(1)
            var bad = False
            for o in range(nops):
                var v = shapes[unsafe_offset=base + o * 4 + d]
                if v < 0:
                    bad = True
                elif dim == 1:
                    dim = v
                elif v != 1 and dim != v:
                    bad = True
            if bad:
                out[unsafe_offset=g * 4 + d] = Int64(-1)
            else:
                out[unsafe_offset=g * 4 + d] = dim


@export("af_broadcast_index")
def af_broadcast_index(
    n: Int, out_dims_addr: Int, in_dims_addr: Int, out_addr: Int
) abi("C"):
    """Flat source offset for every position of a broadcast output.

    The inverse gather map behind `af.broadcast`: for each of the `n` output
    elements, the offset into the operand. The operand is itself column-major,
    so its strides come from its own extents; an axis of extent 1 is the
    broadcast axis and contributes a zero stride, repeating.

    Every extent is hoisted into a local first: reloading them through the
    pointer on each odometer step costs more than the arithmetic does.
    """
    var out_dims = _i64p(out_dims_addr)
    var in_dims = _i64p(in_dims_addr)
    var out = _i64p(out_addr)
    var d0 = out_dims[unsafe_offset=0]
    var d1 = out_dims[unsafe_offset=1]
    var d2 = out_dims[unsafe_offset=2]
    var d3 = out_dims[unsafe_offset=3]
    var s0 = Int64(1)
    var s1 = s0 * in_dims[unsafe_offset=0]
    var s2 = s1 * in_dims[unsafe_offset=1]
    var s3 = s2 * in_dims[unsafe_offset=2]
    var t0 = Int64(0)
    var t1 = Int64(0)
    var t2 = Int64(0)
    var t3 = Int64(0)
    if in_dims[unsafe_offset=0] != 1:
        t0 = s0
    if in_dims[unsafe_offset=1] != 1:
        t1 = s1
    if in_dims[unsafe_offset=2] != 1:
        t2 = s2
    if in_dims[unsafe_offset=3] != 1:
        t3 = s3
    var i0 = Int64(0)
    var i1 = Int64(0)
    var i2 = Int64(0)
    var i3 = Int64(0)
    for t in range(n):
        out[unsafe_offset=t] = i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3
        i0 += 1
        if i0 >= d0:
            i0 = Int64(0)
            i1 += 1
            if i1 >= d1:
                i1 = Int64(0)
                i2 += 1
                if i2 >= d2:
                    i2 = Int64(0)
                    i3 += 1


@export("af_strides4")
def af_strides4(dims_addr: Int, out_addr: Int) abi("C"):
    """Column-major strides of a 4-dimensional extent, `dim4`-style."""
    var dims = _i64p(dims_addr)
    var out = _i64p(out_addr)
    var acc = Int64(1)
    for d in range(4):
        out[unsafe_offset=d] = acc
        acc *= dims[unsafe_offset=d]
