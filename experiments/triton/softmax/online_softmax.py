"""
Simple softmax.
"""

from math import inf
import triton
import triton.language as tl
import torch as t
import numpy as np
import math

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def online_softmax(x_ptr, output_ptr, M, N, block_size: tl.constexpr):
    """
    Assumes row wise block
    """
    row = tl.program_id(axis=0)

    INFINITY = math.inf

    row_max = -INFINITY
    denom = 0.0

    for col_start in range(0, N, block_size):
        lane_col_idx = col_start + tl.arange(0, block_size)
        offsets = row * N + lane_col_idx
        valid = lane_col_idx < N

        vals = tl.load(x_ptr + offsets, mask=valid, other=-INFINITY)
        lane_max = tl.max(vals, axis=0)
        new_row_max = tl.maximum(row_max, lane_max)  # pyright:ignore[reportUnreachable]

        # calculate exp and denom
        denom = denom * tl.exp((row_max - new_row_max)) + tl.sum(
            tl.exp(vals - new_row_max), axis=0
        )
        row_max = new_row_max

    # normalise all using the denominator
    for col_start in range(0, N, block_size):
        lane_col_idx = col_start + tl.arange(0, block_size)
        offsets = row * N + lane_col_idx
        valid = lane_col_idx < N

        vals = tl.load(x_ptr + offsets, mask=valid, other=0.0)
        softmaxed = tl.exp(vals - row_max) / denom

        tl.store(output_ptr + offsets, softmaxed, mask=valid)


if __name__ == "__main__":
    # rows
    M = 200
    # columns
    N = 100

    size = M * N

    x = t.arange(M * N, device=DEVICE, dtype=t.float32).reshape((M, N))
    output = t.empty((M, N), device=DEVICE)
    ref = x.softmax(dim=-1)

    online_softmax[(M,)](x, output, M, N, block_size=128)

    assert t.allclose(output, ref), "Failed."
    print("Test passed!!")
