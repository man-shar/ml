"""Flash Attention 2 implementation using Triton."""

import torch as t
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    softmax_scale,
    M,
    O,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq_len,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq_len,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq_len,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq_len,
    stride_O_dim,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    block_size_q: tl.constexpr,
    block_size_kv: tl.constexpr,
    stage: tl.constexpr,
):
    tl.static_assert(block_size_kv <= head_dim)

    # block of the query block matrix
    block_index_q = tl.program_id(0)

    batch_head_index = tl.program_id(1)

    batch_index = tl.fdiv(batch_head_index, num_heads)

    head_index = batch_head_index % num_heads

    q_block_offset = (
        batch_index.to(tl.int64) * stride_Q_batch
        + head_index.to(tl.int64) * stride_Q_head
    )

    # Q_block_ptr = tl.make_block_ptr(
    #     base=Q + q_block_offset,
    #     shape=(seq_len, head_dim),
    #     strides=(stride_Q_batch, stride_Q_head, stride_Q_seq_len, stride_Q_dim),
    #     offsets=
    #     block_shape=
    # )


class TritonAttention(t.autograd.Function):
    @staticmethod
    def forward(
        ctx, Q: t.tensor, K: t.tensor, V: t.tensor, causal: bool, softmax_scale: float
    ):
        head_dim_q, head_dim_k, head_dim_v = Q.shape[-1], K.shape[-1], V.shape[-1]

        batch_size, num_heads, seq_len, head_dim = Q.shape

        assert head_dim_q == head_dim_k and head_dim_k == head_dim_v

        # This can only be forced to be like Queries, not keys and values
        # because, in cross attention, the encoder is sending in the keys and values
        # while the decoder creates the queries.

        # if query coming from decoder is batch, num_heads, seq_len_decoder, dim
        # and key, values are batch, num_heads, seq_len_encoder, dim
        # then, when doing QK^T, we get:
        # batch, num_heads, seq_len_decoder, dim * batch, num_heads, dim, seq_len_encoder
        # so the output is batch, num_heads, seq_len_decoder, seq_len_encoder
        # then we multipole by v which is batch, num_heads, seq_len_encoder, dim
        # and we get batch, num_heads, seq_len_decoder, dim
        # output is batch_size, num_heads, seq_len, head_dim
        O = t.empty_like(Q)

        stage = 3 if causal else 1

        grid = lambda args: (
            triton.cdiv(
                seq_len, args["BLOCK_SIZE_Q"]
            ),  # which query matrix block aka group of queries we will work with
            batch_size
            * num_heads,  # one GPU block works on all batches and heads for this query block. One thread works on one head of one batch.
            1,
        )

        # number of parallel kernels = batch_size * num_heads * num_blocks_q

        # M is for logsumexp for backward pass
        # stores max of each row + the normalisation factor in one number
        M = t.empty((batch_size, num_heads, seq_len), device=Q.device, dtype=t.float32)

        _attn_fwd(
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride[0],
            stride_Q_head=Q.stride[1],
            stride_Q_seq_len=Q.stride[2],
            stride_Q_dim=Q.stride[3],
            stride_K_batch=K.stride[0],
            stride_K_head=K.stride[1],
            stride_K_seq_len=K.stride[2],
            stride_K_dim=K.stride[3],
            stride_V_batch=V.stride[0],
            stride_V_head=V.stride[1],
            stride_V_seq_len=V.stride[2],
            stride_V_dim=V.stride[3],
            stride_O_batch=O.stride[0],
            stride_O_head=O.stride[1],
            stride_O_seq_len=O.stride[2],
            stride_O_dim=O.stride[3],
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            stage=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.head_dim = head_dim
        ctx.causal = causal

        return O

    @staticmethod
    def backward(ctx):
        pass


def test_op(
    batch_size, seq_len, num_heads, head_dim, causal: bool = False, dtype=t.float16
):
    Q = (
        t.empty((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    K = (
        t.empty((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    V = (
        t.empty((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (head_dim**0.5)

    dO = t.rand_like(Q)

    # reference implementation to test triton outputs against
    mask = t.tril(t.ones((seq_len, seq_len)), device="cuda")

    # this is now batch_size, num_heads, seq_len, seq_len
    P = t.matmul(Q, K.transport(2, 3)) / softmax_scale

    if causal:
        P[:, :, mask == 0] = float("-inf")

    P = t.softmax(P.float(), dim=-1).half()
    # batch_size, num_heads, seq_len, seq_len * batch_size, num_heads, seq_len, head_dim
    # -> batch_size, num_heads, seq_len, head_dim
    ref_O = t.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None

    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None

    rtol = 0.0
    atol = 1e-2

    assert t.allclose(tri_out, ref_O, atol=atol, rtol=rtol)
    assert t.allclose(tri_dK, ref_dK, atol=atol, rtol=rtol)
    assert t.allclose(tri_dQ, ref_dQ, atol=atol, rtol=rtol)
    assert t.allclose(tri_dV, ref_dV, atol=atol, rtol=rtol)
