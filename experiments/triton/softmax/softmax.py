from math import inf
import triton
import triton.language as tl
import torch as t
import numpy as np
import matplotlib.pyplot as plt


@triton.jit
def softmax_simple(
    x_ptr, output_ptr, M, N, stride_x, stride_y, block_size: tl.constexpr
):
    """
    Assumes row wise block
    """
    row = tl.program_id(axis=0)

    if row >= M:
        return

    # M x N matrix
    # we want to loop over this whole row

    # get the max value for this row
    max = -inf
    for col_start in range(0, N, block_size):
        # get the lanes per tile
        offsets = (row * N + col_start) + tl.arange(0, block_size)
        valid = offsets < N
        vals = tl.load(x_ptr + offsets, mask=valid, other=-inf)
        max = tl.max(max, tl.max(vals))

    # now find the denominator in the softmax for this row (sum of exp(x - row_max))
    denom = 0.0
    for col_start in range(0, N, block_size):
        offsets = (row * N + col_start) + tl.arange(0, block_size)
        valid = offsets < N
        vals = tl.load(x_ptr + offsets, mask=valid, other=-inf)
        vals = vals - max

        # store these in the output for now (we will later divide all these by the denom we calculate here)
        tl.store(output_ptr + offsets, vals, mask=valid)

        denom += tl.sum(tl.exp(vals))

    # normalise all using the denominator
    for col_start in range(0, N, block_size):
        offsets = (row * N + col_start) + tl.arange(0, block_size)
        valid = offsets < N

        vals = tl.load(output_ptr + offsets, mask=valid, other=-inf)
        softmaxed = vals / max

        tl.store(output_ptr + offsets, softmaxed, mask=valid)


size = 16_777_216
