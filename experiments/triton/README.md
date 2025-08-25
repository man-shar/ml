Learning triton

Bunch of learnings so far about the library:

1. triton's arange must have constexpr or 0. can't do tl.arange(my_variable, my_variable + XX)
2. triton's dot will only let you dot matrices that have a minimum of 16 rows and 16 columns
3. arange must be a power of 2
4. triton's dot has an accumulator param! awesome!

## Creating tests

Need to be careful about my unit tests.

Assuming a matrix multiplication kernel, if i do:

```python
A = torch.arange(M * N, device=DEVICE, dtype=torch.float32).reshape(M, N)
B = torch.arange(N * K, device=DEVICE, dtype=torch.float32).reshape(N, K)
C = torch.empty((M, K), device=DEVICE, dtype=torch.float32)

ref = A @ B

# ... run kernel

torch.allclose(C, ref)
```

The `allclose` might fail even when the kernel is correct. I need to add some tolerances, but also carefully. If i do:

```python
assert torch.allclose(C, ref, atol=1e-2, rtol=1e-2)
```

This will pass. But it is deceptive. Internally the check is: `error (x - y) <= atol + rtol * y`.

As y grows (like in big `arange`s), the second term is massive. So it passes even if it has a big error.

What I should be _aiming_ to pass is:

```python
assert torch.allclose(C, ref, atol=1e-2, rtol=0)
```

But it might fail again! This time, _because_ of the big aranges that were saving us earlier! And _not_ because of the tolerances. (seems like it has to do with something called ULP of floats. [ChatGPT chat here](https://chatgpt.com/share/e/68ace452-bac4-800a-9b51-d659b11f46d4), stack overflow answer [here](https://stackoverflow.com/questions/43965347/ulp-unit-of-least-precision). Not entirely clear to me yet.)

Seems like this is best thing to do when initialising my inputs:

```python
A = torch.rand((M, N), device=DEVICE, dtype=torch.float32) - 0.5
B = torch.rand((N, K), device=DEVICE, dtype=torch.float32) - 0.5
```

I guess that:

- brings the numbers from -0.5 to 0.5. So things.. cancel out?
- Also removes the relative tolerance term completely. So tiny numbers, tiny absolute tolerance = pass.
- The absolute tolerances
