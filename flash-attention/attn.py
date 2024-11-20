"""Flash Attention 2 implementation using Triton."""

import torch as t
import triton
import triton.language as tl


class TritonAttention(t.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        pass

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
