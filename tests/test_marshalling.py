"""Tests for the host-side marshalling and broadcast kernels of mojo_arrayfire.

`reorder2d` is checked against NumPy's own layout arithmetic,
`broadcast_index` against what `np.broadcast_to` selects, and
`broadcast_shape` against `np.broadcast_shapes`. Every quantity is an integer
index or an exact data move, so every comparison is exact.

`arrayfire.array._ctype_to_lists` is deliberately absent: for the layouts
ArrayFire actually produces, its index map is the identity, so there is no
arithmetic to compile. See the README.
"""

import numpy as np
import pytest

import mojo_arrayfire as maf


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1, 5), (5, 1), (7, 4), (4, 7)])
def test_reorder2d_is_the_fortran_image(shape):
    rng = np.random.default_rng(1)
    src = rng.standard_normal(shape)
    got = maf.reorder2d(src)
    assert np.array_equal(got, src)
    assert got.flags["F_CONTIGUOUS"]
    assert maf.reorder2d(src).reshape(-1, order="F").tolist() == \
        src.reshape(-1, order="F").tolist()


def test_reorder2d_matches_the_cc_to_af_map():
    """`dst[i0 + i1*n0] = src[i1 + i0*n1]`, applied to the flat buffer."""
    n0, n1 = 2, 3
    src = np.array([[0, 1, 2], [3, 4, 5]])
    got = maf.reorder2d(src)
    flat = got.reshape(-1, order="F")
    expect = []
    for i1 in range(n1):
        for i0 in range(n0):
            expect.append(int(src.reshape(-1)[i1 + i0 * n1]))
    assert flat.tolist() == expect


def test_reorder2d_rejects_non_2d():
    with pytest.raises(ValueError):
        maf.reorder2d(np.zeros(5))


def test_broadcast_shape_matches_numpy():
    groups = [
        ((2, 3, 1, 1), (1, 3, 4, 1)),
        ((5, 1, 7, 1), (1, 4, 7, 2)),
        ((3, 3, 3, 3), (1, 1, 1, 1)),
        ((1, 1, 1, 1), (9, 8, 1, 1)),
        ((2, 1, 3, 1), (1, 4, 3, 5)),
    ]
    got = maf.broadcast_shape(groups)
    for i, g in enumerate(groups):
        expect = np.broadcast_shapes(*g)
        assert tuple(got[i]) == tuple(expect), f"group {i}"


def test_broadcast_shape_rejects_ragged_operand_counts():
    with pytest.raises(ValueError):
        maf.broadcast_shape([((1, 1, 1, 1),), ((1, 1, 1, 1), (2, 1, 1, 1))])
    with pytest.raises(ValueError):
        maf.broadcast_shape([((1, 1, 1),)])


def test_broadcast_shape_rejects_incompatible():
    with pytest.raises(ValueError):
        maf.broadcast_shape([((2, 3, 1, 1), (4, 1, 1, 1))])
    with pytest.raises(ValueError):
        maf.broadcast_shape([((2, 1, 1, 1),), ((2, 1, 1, 1), (3, 1, 1, 1))])
    with pytest.raises(ValueError):
        maf.broadcast_shape([((2, 3),)])


def test_broadcast_shape_batch_is_elementwise():
    groups = [((2, 3, 1, 1), (1, 3, 4, 1)), ((6, 1, 1, 1), (1, 1, 7, 1))]
    assert maf.broadcast_shape(groups).tolist() == [[2, 3, 4, 1], [6, 1, 7, 1]]


@pytest.mark.parametrize(
    "out_dims,in_dims",
    [
        ((2, 3, 1, 1), (1, 3, 1, 1)),
        ((2, 3, 4, 1), (2, 1, 1, 1)),
        ((2, 3, 1, 1), (1, 1, 1, 1)),
        ((3, 1, 5, 1), (1, 1, 5, 1)),
        ((3, 1, 1, 1), (1, 1, 1, 1)),
        ((1, 1, 1, 1), (1, 1, 1, 1)),
        ((4, 4, 1, 2), (1, 4, 1, 1)),
    ],
)
def test_broadcast_index_gathers_like_numpy(out_dims, in_dims):
    """The offsets must select the same elements `np.broadcast_to` would."""
    got = maf.broadcast_index(out_dims, in_dims)
    src = np.arange(int(np.prod(in_dims)), dtype=np.int64)
    expanded = np.broadcast_to(src.reshape(in_dims, order="F"), out_dims)
    expect = expanded.reshape(-1, order="F")
    assert src[got].tolist() == [int(v) for v in expect]
    assert got.min() >= 0 and got.max() < src.size


def test_broadcast_index_repeated_axis_gets_a_zero_stride():
    """A source extent of 1 must contribute stride 0, not the output stride."""
    got = maf.broadcast_index((3, 1, 1, 1), (1, 1, 1, 1))
    assert got.tolist() == [0, 0, 0]
    assert maf.broadcast_index((2, 2, 1, 1), (2, 1, 1, 1)).tolist() == [0, 1, 0, 1]
    assert maf.broadcast_index((1, 3, 1, 1), (1, 3, 1, 1)).tolist() == [0, 1, 2]
    assert maf.broadcast_index((2, 1, 1, 1), (1, 1, 1, 1)).tolist() == [0, 0]


def test_broadcast_index_rejects_bad_arity():
    with pytest.raises(ValueError):
        maf.broadcast_index((2, 3), (2, 1))
    with pytest.raises(ValueError):
        maf.broadcast_index((2, 3, 1, 1), (2, 1, 1))


def test_strides4_is_column_major():
    assert maf.strides4((2, 3, 4, 5)).tolist() == [1, 2, 6, 24]
    assert maf.strides4((1, 1, 1, 1)).tolist() == [1, 1, 1, 1]
    with pytest.raises(ValueError):
        maf.strides4((2, 3))
