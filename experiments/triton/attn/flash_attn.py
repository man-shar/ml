"""
Flash Attention
"""

import triton
import triton.language as tl
import torch

from experiments.triton.utils import DEVICE


@triton.jit
def flash_attn(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    seq_len: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    causal: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
):
    """
    Q is seq x head_dim
    K is seq x head_dim
    Q x K^T gives seq x seq (attn_scores)

    softmax(Q x K^T) gives seq x seq (attn_probs)
    V is seq x head_dim

    softmax(Q.K^T) x V gives seq x head_dim

    store the above into O

    causal = 1 or 0 (masks "future" tokens if 1)

    BLOCK_Q is the number of rows this program owns.

    BLOCK_KV defines the chunks in which we will load rows of K and V for the attention calculation.
    """
    i = tl.program_id(0)

    start_q = i * BLOCK_Q  # index of the first row this program will own

    # load start_q -> start_q + BLOCK_Q rows from Q
    # we load a [BLOCK_Q x head_dim] tensor
    Q_col_offsets = tl.arange(0, head_dim)[None, :]  # [1, head_dim]
    # we don't need a mask for this
    Q_col_mask = True

    Q_row_offsets = (head_dim * (start_q + tl.arange(0, BLOCK_Q)))[
        :, None
    ]  # [BLOCK_Q, 1]
    Q_row_mask = (start_q + tl.arange(0, BLOCK_Q))[:, None] < seq_len

    Q_offsets = Q_row_offsets + Q_col_offsets
    Q_mask = Q_col_mask & Q_row_mask

    Q_vals = tl.load(Q_ptr + Q_offsets, mask=Q_mask, other=0.0).to(
        tl.float32
    )  # [BLOCK_Q, head_dim]

    O_accumulator = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)

    row_maxes = tl.full((BLOCK_Q, 1), float("-inf"), dtype=tl.float32)
    row_sums = tl.full((BLOCK_Q, 1), 0, dtype=tl.float32)

    # we now iterate over rows of K and V in BLOCK_KV chunks
    for start_kv in tl.range(0, seq_len, BLOCK_KV):
        # load start_kv -> start_kv + BLOCK_KV rows from K
        # we load a [BLOCK_KV x head_dim] tensor
        K_col_offsets = tl.arange(0, head_dim)[None, :]  # [1, head_dim]
        # we don't need a mask for this
        K_col_mask = True

        K_row_offsets = (head_dim * (start_kv + tl.arange(0, BLOCK_KV)))[
            :, None
        ]  # [BLOCK_KV, 1]
        K_row_mask = (start_kv + tl.arange(0, BLOCK_KV))[:, None] < seq_len

        K_offsets = K_row_offsets + K_col_offsets
        K_mask = K_col_mask & K_row_mask

        K_vals = tl.load(K_ptr + K_offsets, mask=K_mask, other=0.0).to(
            tl.float32
        )  # [BLOCK_KV, head_dim]

        # we will multiply Q and K to get a tensor of size [BLOCK_Q x BLOCK_KV]
        QK = tl.dot(Q_vals, tl.trans(K_vals)) * sm_scale  # [BLOCK_Q, BLOCK_KV]

        # get the row maxes of these along the columns
        block_maxes = tl.max(QK, axis=1, keep_dims=True)  # [BLOCK_Q, 1]

        new_maxes = tl.maximum(row_maxes, block_maxes)

        # this is what we will multiply all older values by
        scale = tl.exp(row_maxes - new_maxes)

        row_maxes = new_maxes

        block_sums = tl.sum(
            tl.exp(QK - new_maxes), axis=1, keep_dims=True
        )  # [BLOCK_Q, 1]

        row_sums = row_sums * scale + block_sums

        # we calculate a partial softmax (no division. we do it at the end with our final scores and maxes)
        partial_softmax = tl.exp((QK - new_maxes))  # [BLOCK_Q, BLOCK_KV]

        # multiply by V's current block
        V_col_offsets = tl.arange(0, head_dim)[None, :]  # [1, head_dim]
        # we don't need a mask for this
        V_col_mask = True

        V_row_offsets = (head_dim * (start_kv + tl.arange(0, BLOCK_KV)))[
            :, None
        ]  # [BLOCK_KV, 1]
        V_row_mask = (start_kv + tl.arange(0, BLOCK_KV))[:, None] < seq_len

        V_offsets = V_row_offsets + V_col_offsets
        V_mask = V_col_mask & V_row_mask

        V_vals = tl.load(V_ptr + V_offsets, mask=V_mask, other=0.0).to(
            tl.float32
        )  # [BLOCK_KV, head_dim]

        block_O = tl.dot(partial_softmax, V_vals)  # [BLOCK_Q, head_dim]

        O_accumulator = O_accumulator * scale + block_O

    O_accumulator = O_accumulator / row_sums  # [BLOCK_Q, head_dim]

    O_row_offsets = (
        head_dim * (start_q + tl.arange(0, BLOCK_Q))[:, None]
    )  # [BLOCK_Q, 1]
    O_row_mask = ((start_q + tl.arange(0, BLOCK_Q)) < seq_len)[:, None]

    O_col_offsets = tl.arange(0, head_dim)[None, :]  # [head_dim, 1]
    O_col_mask = True

    O_offsets = O_row_offsets + O_col_offsets
    O_mask = O_row_mask & O_col_mask

    tl.store(O_ptr + O_offsets, O_accumulator, mask=O_mask)


def bench_runner(
    seq_len: int = 100,
    head_dim: int = 128,
    block_size: int = 32,
    BLOCK_KV: int = 16,
    num_warps: int = 2,
    dtype: torch.dtype = torch.float32,
    causal: bool = False,
):
    Q = torch.rand((seq_len, head_dim), dtype=dtype, device=DEVICE) - 0.5
    K = torch.rand((seq_len, head_dim), dtype=dtype, device=DEVICE) - 0.5
    V = torch.rand((seq_len, head_dim), dtype=dtype, device=DEVICE) - 0.5
    O = torch.empty((seq_len, head_dim), dtype=dtype, device=DEVICE)

    grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_Q"]),)

    flash_attn[grid](
        Q,
        K,
        V,
        O,
        seq_len=seq_len,
        BLOCK_Q=block_size,
        BLOCK_KV=BLOCK_KV,
        causal=causal,
        num_warps=num_warps,
        head_dim=head_dim,
        sm_scale=(1.0 / (head_dim**0.5)),
    )

    return Q, K, V, O


if __name__ == "__main__":
    head_dim = 128
    sm_scale = 1.0 / (head_dim**0.5)

    Q, K, V, O = bench_runner(head_dim=128)

    with torch.no_grad():
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q.unsqueeze(0).unsqueeze(0),  # Add batch and head dims
            K.unsqueeze(0).unsqueeze(0),
            V.unsqueeze(0).unsqueeze(0),
            scale=1.0 / (head_dim**0.5),
        ).squeeze()

    max_diff = torch.max(ref - O)
    mean_diff = torch.mean(ref - O)
    diff_str = f"Max difference: {max_diff}, Mean diff: {mean_diff}"

    assert torch.allclose(ref, O, atol=1e-2, rtol=0), (
        f"Tests failed. Did you use float32? Flash attention is known to be unstable on float 32. {diff_str}"
    )

    print(f"All tests passed!!! {diff_str}")
