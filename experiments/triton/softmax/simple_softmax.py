"""
Simple softmax.
"""

import math
import triton
import triton.language as tl
import torch as t

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def simple_softmax(x_ptr, output_ptr, M, N, block_size: tl.constexpr):
    """
    Assumes row wise block
    """
    row = tl.program_id(axis=0)

    # get the max value for this row
    INFINITY = math.inf

    row_max = -INFINITY
    for col_start in range(0, N, block_size):
        lane_col_idx = col_start + tl.arange(0, block_size)
        offsets = row * N + lane_col_idx
        valid = lane_col_idx < N

        vals = tl.load(x_ptr + offsets, mask=valid, other=-INFINITY)
        lane_max = tl.max(vals, axis=0)
        row_max = tl.maximum(row_max, lane_max)  # pyright:ignore[reportUnreachable]

    # now find the denominator in the softmax for this row (sum of exp(x - row_max))
    denom = 0.0
    for col_start in range(0, N, block_size):
        # get the vector lanes of this tile
        lane_col_idx = col_start + tl.arange(0, block_size)
        offsets = row * N + lane_col_idx
        valid = lane_col_idx < N

        vals = tl.load(x_ptr + offsets, mask=valid, other=-INFINITY)
        vals = tl.exp(vals - row_max)

        # store these in the output for now (we will later divide all these by the denom we calculate here)
        tl.store(output_ptr + offsets, vals, mask=valid)

        denom += tl.sum(vals, axis=0)

    # normalise all using the denominator
    for col_start in range(0, N, block_size):
        lane_col_idx = col_start + tl.arange(0, block_size)
        offsets = row * N + lane_col_idx
        valid = lane_col_idx < N

        # note we read from output not x here (we saved exps in output above)
        vals = tl.load(output_ptr + offsets, mask=valid, other=0.0)
        softmaxed = vals / denom

        tl.store(output_ptr + offsets, softmaxed, mask=valid)


def bench_runner(M: int, N: int, block_size: int, dtype: t.dtype, num_warps: int):
    """
    Run the benchmark for a given M, N, block_size, dtype.
    """

    x = t.arange(M * N, device=DEVICE, dtype=dtype).reshape((M, N))
    output = t.empty((M, N), device=DEVICE)

    simple_softmax[(M,)](x, output, M, N, block_size=block_size)

    return x, output


if __name__ == "__main__":
    # rows
    M = 200
    # columns
    N = 100
    x, output = bench_runner(M, N, 128, t.float32)
    ref = x.softmax(dim=-1)

    assert t.allclose(output, ref), "Failed."
    print("Test passed!!")
