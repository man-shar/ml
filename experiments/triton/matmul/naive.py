"""
Naive matmul where one program handles one element of the output matrix.
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# A: MxN matrix
# B: NxK matrix (we do A @ B)
# C: MxK matrix
@triton.jit
def naive_matmul(A_ptr, B_ptr, C_ptr, M, N, K, BLOCK_N: tl.constexpr):
    """
    One program handles one item of the output matrix C
    """
    i = tl.program_id(0)
    j = tl.program_id(1)

    if i >= M or j >= K:
        return

    # this one program will need: one row of A and one column of B
    # we need: A's i'th row
    # and one column of B for that row

    acc = 0.0
    for start in range(0, N, BLOCK_N):
        A_offsets = start + tl.arange(0, BLOCK_N)
        valid = A_offsets < N

        # this has N values
        A_vals = tl.load(A_ptr + A_offsets, mask=valid, other=0)

        # get the next block_n elements of the j'th *column* of B (which will also have N values)
        # the offsets here are total garbage with no coalescing.
        # every value in a column of B is separated by K (because they are in separate rows and B has K columns)
        # this gives us 0 * k, 1 * k, 2k.... BLOCK_N * k (this is the first element of each row)
        # b offsets and a offsets are the same
        B_offsets = A_offsets * K
        # we now offset it by j to give the jth column of each row
        B_offsets = B_offsets + j
        B_vals = tl.load(B_ptr + B_offsets, mask=valid, other=0)

        # get product and sum
        acc += tl.sum(A_vals * B_vals, axis=0)  # pyright:ignore[reportUnreachable]
        # add this to the acumulator

    C_offset = C_ptr + i * K + j
    tl.store(C_offset, acc)


if __name__ == "__main__":
    M = 200
    N = 100
    K = 30

    BLOCK_N = 128

    A = torch.rand((M, N), device=DEVICE, dtype=torch.float32)
    B = torch.rand((N, K), device=DEVICE, dtype=torch.float32)
    C = torch.empty((M, K), device=DEVICE, dtype=torch.float32)

    ref = A @ B

    naive_matmul[(M, K)](A, B, C, M, N, K, BLOCK_N)

    torch.allclose(C, ref), "Tests failed"

    print("All tests passed!!!")


def bench_runner(
    M: int, N: int, C: int, block_size: int, dtype: torch.dtype, num_warps: int
):
    """
    Run the benchmark for a given M, N, block_size, dtype.
    """
    A = torch.rand((M, N), device=DEVICE, dtype=dtype)
    B = torch.rand((N, K), device=DEVICE, dtype=dtype)
    output = torch.empty((M, K), device=DEVICE, dtype=dtype)

    naive_matmul[(M, K)](A, B, C, M, N, K, BLOCK_N=block_size)

    return A, B, C, output
