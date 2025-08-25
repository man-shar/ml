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
def naive_matmul(
    A_ptr, B_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    """
    One program handles one item of the output matrix C
    """
    i = tl.program_id(0)
    j = tl.program_id(1)

    if i >= M:
        return

    # this one program will need: one row of A and one column of B
    # we need: A's i'th row
    # and one column of B for that row

    A_offsets = i * N + tl.arange(0, N)
    # this has N values
    A_vals = tl.load(A_ptr + A_offsets)

    # get the j'th *column* of B (which will also have N values)
    # the offsets here are total garbage with no coalescing.
    # every value in a column of B is separated by K (because they are in separate rows and B has K columns)

    # this gives us 0 * k, 1 * k, 2k.... N * k (this is the first element of each row)
    B_offsets = tl.arange(0, N) * K
    # we now offset it by j
    B_offsets = B_offsets + j
    B_vals = tl.load(B_ptr + B_offsets)

    # get product and sum
    output = tl.sum(A_vals * B_vals, axis=0)  # pyright:ignore[reportUnreachable]

    C_offset = C_ptr + i * K + j
    tl.store(C_offset, output)


if __name__ == "__main__":
    # we are limited to having powers of 2 here because we use these inside triton's arange which forces us to have powers of 2
    M = 32
    N = 16
    K = 8

    A = torch.rand((M, N), device=DEVICE, dtype=torch.float32)
    B = torch.rand((N, K), device=DEVICE, dtype=torch.float32)
    C = torch.empty((M, K), device=DEVICE, dtype=torch.float32)

    ref = A @ B

    naive_matmul[(M, K)](A, B, C, M, N, K)

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

    naive_matmul[(M, K)](A, B, C, M, N, K)

    return A, B, C, output
