"""
Blocked matrix multiplication.
"""

from math import ceil
import triton
import triton.language as tl
import torch
from ..utils import DEVICE


# A is MxN matrix
# B is NxK matrix
# C is a MxK matrix
@triton.jit
def blocked_matmul(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    OUTPUT_BLOCK_M: tl.constexpr,
    OUTPUT_BLOCK_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    OUTPUT_BLOCK_M x OUTPUT_BLOCK_K is the output's tile we will calculate in this program.

    We will load full rows and columns of A and B, but in CHUNK_SIZE chunks.
    """
    # i, j of this program tile
    i = tl.program_id(0)
    j = tl.program_id(1)

    # the accumulator block which we will write to the output matrix eventually
    accumulator = tl.zeros((OUTPUT_BLOCK_M, OUTPUT_BLOCK_K), dtype=tl.float32)

    row_start = i * OUTPUT_BLOCK_M
    col_start = j * OUTPUT_BLOCK_K

    # this blnock of the output matrix is formed by:
    # - rows of A from row_start to row_start + OUTPUT_BLOCK_M
    # - columns of B from col_start to col_start + OUTPUT_BLOCK_K

    # each row of A has N columns
    for A_col_start in range(0, N, CHUNK_SIZE):
        # get the column offsets of A. basically A_col_start to A_col_start + CHUNK_SIZE
        A_col_offsets = (
            A_col_start + tl.arange(0, CHUNK_SIZE)[None, :]
        )  # [1, CHUNK_SIZE]
        A_col_mask = A_col_offsets < N  # [1, CHUNK_SIZE]

        # each row's 0 index are: row_start, row_start + 1 * N, row_start + 2 * N...
        A_row_offsets = (
            N * (row_start + tl.arange(0, OUTPUT_BLOCK_M))[:, None]
        )  # [OUTPUT_BLOCK_M, 1]

        A_row_mask = ((row_start + tl.arange(0, OUTPUT_BLOCK_M)) < M)[
            :, None
        ]  # [OUTPUT_BLOCK_M, 1]

        # sum. this broadcasts and magically creates an [OUTPUT_BLOCK_M, CHUNK_SIZE] matrix
        A_offsets = A_row_offsets + A_col_offsets  # [OUTPUT_BLOCK_M, CHUNK_SIZE]
        A_mask = A_col_mask & A_row_mask

        A_vals = tl.load(
            A_ptr + A_offsets, mask=A_mask, other=0
        )  # [OUTPUT_BLOCK_M, CHUNK_SIZE]

        # do the same for B
        # for B we want the CHUNK_SIZE rows from col_start to col_start + OUTPUT_BLOCK_K
        # where the rows in each of those columns we get go from A_col_start to A_col_start + CHUNK_SIZE
        B_col_offsets = (
            col_start + tl.arange(0, OUTPUT_BLOCK_K)[None, :]
        )  # [1, OUTPUT_BLOCK_K]
        B_col_mask = B_col_offsets < K

        # each row of B is non contiguous indices, each separated by K
        B_row_offsets = (
            K * (A_col_start + tl.arange(0, CHUNK_SIZE))[:, None]
        )  # [CHUNK_SIZE, 1]
        B_row_mask = ((A_col_start + tl.arange(0, CHUNK_SIZE)) < N)[:, None]

        B_offsets = B_row_offsets + B_col_offsets  # [CHUNK_SIZE, OUTPUT_BLOCK_K]
        B_mask = B_col_mask & B_row_mask

        B_vals = tl.load(
            B_ptr + B_offsets, mask=B_mask, other=0
        )  # [CHUNK_SIZE, OUTPUT_BLOCK_K]

        accumulator = tl.dot(
            A_vals, B_vals, accumulator
        )  # [OUTPUT_BLOCK_M, OUTPUT_BLOCK_K]

    C_row_offsets = (K * (row_start + tl.arange(0, OUTPUT_BLOCK_M)))[
        :, None
    ]  # [OUTPUT_BLOCK_M, 1]
    C_row_mask = ((row_start + tl.arange(0, OUTPUT_BLOCK_M)) < M)[
        :, None
    ]  # [OUTPUT_BLOCK_M, 1]

    C_col_offsets = (col_start + tl.arange(0, OUTPUT_BLOCK_K))[
        None, :
    ]  # [1, OUTPUT_BLOCK_K]
    C_col_mask = C_col_offsets < K

    C_offsets = C_row_offsets + C_col_offsets
    C_mask = C_row_mask & C_col_mask

    tl.store(C_ptr + C_offsets, accumulator, mask=C_mask)


def bench_runner(
    M: int,
    N: int,
    K: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
    num_warps: int = 2,
    OUTPUT_BLOCK_M=16,
    OUTPUT_BLOCK_K=16,
):
    """
    Run the benchmark for a given M, N, block_size, dtype.
    """
    A = torch.rand((M, N), device=DEVICE, dtype=dtype) - 0.5
    B = torch.rand((N, K), device=DEVICE, dtype=dtype) - 0.5
    C = torch.empty((M, K), device=DEVICE, dtype=dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["OUTPUT_BLOCK_M"]),
        triton.cdiv(K, meta["OUTPUT_BLOCK_K"]),
    )

    blocked_matmul[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        OUTPUT_BLOCK_M=OUTPUT_BLOCK_M,
        OUTPUT_BLOCK_K=OUTPUT_BLOCK_K,
        CHUNK_SIZE=block_size,
        num_warps=num_warps,
    )

    return A, B, C


if __name__ == "__main__":
    M = 100
    N = 200
    K = 300

    A, B, C = bench_runner(
        M,
        N,
        K,
    )

    ref = A @ B

    assert torch.allclose(ref, C, atol=1e-2), "Tests failed"

    print("All tests passed!!!")
