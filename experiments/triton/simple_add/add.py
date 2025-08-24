# simple add

import time
import torch

import triton
import triton.language as tl
import numpy as np
from plot_results import plot


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)

    valid = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=valid, other=0)
    y = tl.load(y_ptr + offsets, mask=valid, other=0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=valid)


def driver(
    x: torch.Tensor,
    y: torch.Tensor,
    output_triton: torch.Tensor,
    size: int,
    block_size: int,
    num_warps: int,
    iters: int = 10,
) -> float:
    grid = lambda meta: (triton.cdiv(size, meta["block_size"]),)

    output_torch = x + y
    torch.cuda.synchronize()

    # 2) Warm up this exact specialization (JIT + caches)
    add_kernel[grid](x, y, output_triton, size, block_size=block_size)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iters):
        add_kernel[grid](x, y, output_triton, size, block_size=block_size)

    end.record()

    end.synchronize()

    ms = start.elapsed_time(end) / iters

    assert torch.allclose(output_torch, output_triton)

    return ms


dtypes = [torch.float16, torch.bfloat16, torch.float32]
dtype_bytes = [2, 2, 4]
block_sizes = [128, 256, 512, 1024, 2048]
num_warps_list = [1, 2, 4, 8]

n_elements = 16_777_216

results = ["dtype,block_size,num_warps,ms,total_size_in_gb,gbps"]


combinations = []

for dtype, dtype_size_in_bytes in zip(dtypes, dtype_bytes):
    for bs in block_sizes:
        for num_warps in num_warps_list:
            combinations.append([dtype, dtype_size_in_bytes, bs, num_warps])

np.random.shuffle(combinations)

for dtype, dtype_size_in_bytes in zip(dtypes, dtype_bytes):
    x = torch.rand(n_elements, device=DEVICE, dtype=dtype)
    y = torch.rand(n_elements, device=DEVICE, dtype=dtype)
    output_triton = torch.empty(n_elements, device=DEVICE, dtype=dtype)

    for bs in block_sizes:
        for num_warps in num_warps_list:
            ms = driver(x, y, output_triton, n_elements, bs, num_warps=num_warps)

            total_size_in_gb = 3 * n_elements * dtype_size_in_bytes / 1e9

            gbps = total_size_in_gb / (ms / 1e3)

            results.append(
                ",".join(
                    [
                        str(dtype),
                        str(bs),
                        str(num_warps),
                        str(ms),
                        str(total_size_in_gb),
                        str(gbps),
                    ]
                )
            )

open("results.csv", "w").write("\n".join(results))

plot()
