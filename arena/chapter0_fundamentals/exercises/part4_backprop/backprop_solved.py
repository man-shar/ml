# %% [markdown]
# # [0.4] - Build Your Own Backpropagation Framework
#

# %% [markdown]
# Colab: [exercises](https://colab.research.google.com/drive/1el3ba9T6ORczG7prKzKe1iKZnR9xspYz) | [solutions](https://colab.research.google.com/drive/19fhocNzLbCmYDOsMKn9h8Ps7yTwnKNVN)
#
# ARENA 3.0 [Streamlit page](https://arena3-chapter0-fundamentals.streamlit.app/[0.4]_Backprop)
#
# Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-uk/shared_invite/zt-2noug8mpy-TRYbCnc3pzj7ITNrZIjKww), and ask any questions on the dedicated channels for this chapter of material.
#

# %% [markdown]
# <img src="https://raw.githubusercontent.com/callummcdougall/Fundamentals/main/images/backprop.png" width="350">
#

# %% [markdown]
# # Introduction

# %% [markdown]
# Today you're going to build your very own system that can run the backpropagation algorithm in essentially the same way as PyTorch does. By the end of the day, you'll be able to train a multi-layer perceptron neural network, using your own backprop system!
#
# The main differences between the full PyTorch and our version are:
#
# * We will focus on CPU only, as all the ideas are the same on GPU.
# * We will use NumPy arrays internally instead of ATen, the C++ array type used by PyTorch. Backpropagation works independently of the array type.
# * A real `torch.Tensor` has about 700 fields and methods. We will only implement a subset that are particularly instructional and/or necessary to train the MLP.
#
# Note - for today, I'd lean a lot more towards being willing to read the solutions, and even move on from some of them if you don't fully understand them. The low-level messy implementation details for today are much less important than the high-level conceptual takeaways.
#
# Also, if you don't have enough time to finish all sections (which is understandable, because there's a *lot* of content today!), I'd focus on sections **Introduction** and **Autograd**, since conceptually these are the most important. Once you've done both of these, you should have a strong working understanding of the mechanics of backpropagation. If you've finished these sections but you still have some time, others I'd recommend taking a closer look at the backwards functions for matrix multiplication (at the end of section 3) and the `NoGrad` context manager (near the end of section 4).

# %% [markdown]
# ## Content & Learning Objectives
#

# %% [markdown]
# ### 1️⃣ Introduction to backprop
#
# This takes you through what a **computational graph** is, and the basics of how gradients can be backpropagated through such a graph. You'll also implement the backwards versions of some basic functions: if we have tensors `output = func(input)`, then the backward function of `func` can calculate the grad of `input` as a function of the grad of `output`.
#
# > ##### Learning Objectives
# >
# > * Understand what a computational graph is, and how it can be used to calculate gradients.
# > * Start to implement backwards versions of some basic functions.
#
# ### 2️⃣ Autograd
#
# This section goes into more detail on the backpropagation methodology. In order to find the `grad` of each tensor in a computational graph, we first have to perform a **topological sort** of the tensors in the graph, so that each time we try to calculate `tensor.grad`, we've already computed all the other gradients which are used in this calculation. We end this section by writing a `backprop` function, which works just like the `tensor.backward()` method you're already used to in PyTorch.
#
# > ##### Learning Objectives
# >
# > * Perform a topological sort of a computational graph (and understand why this is important).
# > * Implement a the `backprop` function, to calculate and store gradients for all tensors in a computational graph.
#
# ### 3️⃣ More forward & backward functions
#
# Now that we've established the basics, this section allows you to add more forward and backward functions, extending the set of functions we can use in our computational graph.
#
# > ##### Learning Objectives
# >
# > * Implement more forward and backward functions, including for
# >   * Indexing
# >   * Non-differentiable functions
# >   * Matrix multiplication
#
# ### 4️⃣ Putting everything together
#
# In this section, we build your own equivalents of `torch.nn` features like `nn.Parameter`, `nn.Module`, and `nn.Linear`. We can then use these to build our own neural network to classify MINST data.
#
# This completes the chain which starts at basic numpy arrays, and ends with us being able to build essentially any neural network architecture we want!
#
# > ##### Learning Objectives
# >
# > * Complete the process of building up a neural network from scratch and training it via gradient descent.
#
# ### 5️⃣ Bonus
#
# A few bonus exercises are suggested, for pushing your understanding of backpropagation further.
#

# %% [markdown]
# ## Setup (don't read, just run!)
#

# %%

# # Install packages
# %pip install einops
# %pip install jaxtyping

# # Code to download the necessary files (e.g. solutions, test funcs)
# import os, sys
# if not os.path.exists("chapter0_fundamentals"):
#     !wget https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/main.zip
#     !unzip /content/main.zip 'ARENA_3.0-main/chapter0_fundamentals/exercises/*'
#     os.remove("/content/main.zip")
#     os.rename("ARENA_3.0-main/chapter0_fundamentals", "chapter0_fundamentals")
#     os.rmdir("ARENA_3.0-main")
#     sys.path.insert(0, "chapter0_fundamentals/exercises")

# # Clear output
# from IPython.display import clear_output
# clear_output()
# print("Imports & installations complete!")

# %%
import os
import sys
import re
import time
import torch as t
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Iterable, Optional, Union, Dict, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

Arr = np.ndarray
grad_tracking_enabled = True


# Get file paths to this set of exercises
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_backprop"

if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
import part4_backprop.tests as tests
from part4_backprop.utils import visualize, get_mnist
from plotly_utils import line

MAIN = __name__ == "__main__"

# %% [markdown]
# <details>
# <summary>Help - I get a NumPy-related error</summary>
#
# This is an annoying colab-related issue which I haven't been able to find a satisfying fix for. If you restart runtime (but don't delete runtime), and run just the imports cell above again (but not the `%pip install` cell), the problem should go away.
# </details>
#

# %% [markdown]
# # 1️⃣ Introduction to backprop
#

# %% [markdown]
# > ## Learning Objectives
# >
# > * Understand what a computational graph is, and how it can be used to calculate gradients.
# > * Start to implement backwards versions of some basic functions.
#

# %% [markdown]
# ## Reading
#
# * [Calculus on Computational Graphs: Backpropagation (Chris Olah)](https://colah.github.io/posts/2015-08-Backprop/)
#

# %% [markdown]
# ## Computing Gradients with Backpropagation
#
# This section will briefly review the backpropagation algorithm, but focus mainly on the concrete implementation in software.
#
# To train a neural network, we want to know how the loss would change if we slightly adjust one of the learnable parameters.
#
# One obvious and straightforward way to do this would be just to add a small value  to the parameter, and run the forward pass again. This is called finite differences, and the main issue is we need to run a forward pass for every single parameter that we want to adjust. This method is infeasible for large networks, but it's important to know as a way of sanity checking other methods.
#
# A second obvious way is to write out the function for the entire network, and then symbolically take the gradient to obtain a symbolic expression for the gradient. This also works and is another thing to check against, but the expression gets quite complicated.
#
# Suppose that you have some **computational graph**, and you want to determine the derivative of the some scalar loss L with respect to NumPy arrays a, b, and c:
#

# %% [markdown]
# <img src="https://raw.githubusercontent.com/callummcdougall/Fundamentals/main/images/abc_de_L.png" width="400">
#

# %% [markdown]
# This graph corresponds to the following Python:
#
# ```
# d = a * b
# e = b + c
# L = d + e
# ```
#
# The goal of our system is that users can write ordinary looking Python code like this and have all the book-keeping needed to perform backpropagation happen behind the scenes. To do this, we're going to wrap each array and each function in special objects from our library that do the usual thing plus build up this graph structure that we need.
#

# %% [markdown]
# ### Backward Functions
#
# We've drawn our computation graph from left to right and the arrows pointing to the right, so that in the forward pass, boxes to the right depend on boxes to the left. In the backwards pass, the opposite is true: the gradient of boxes on the left depends on the gradient of boxes on the right.
#
# If we want to compute the derivative of $L$ wrt all other variables (as was described in the reading), we should traverse the graph from right to left. Each time we encounter an instance of function application, we can use the chain rule from calculus to proceed one step further to the left. For example, if we have $d = a \times b$, then:
#
# $$
# \frac{dL}{da} = \frac{dL}{dd}\times \frac{dd}{da} = \frac{dL}{dd}\times b
# $$
#
# Suppose we are working from right to left, trying to calculate $\frac{dL}{da}$. If we already know the values of the variables $a$, $b$ and $d$, as well as the value of $\frac{dL}{dd}$, then we can use the following function to find $\frac{dL}{da}$:
#
# $$
# F(a, b, d, \frac{\partial L}{\partial d}) = \frac{\partial L}{\partial d}\cdot b
# $$
#
# and we can do something similar when trying to calculate $\frac{dL}{db}$.
#
# In other words, we can take the **"forward function"** $(a, b) \to a \cdot b$, and for each of its parameters, we can define an associated **"backwards function"** which tells us how to compute the gradient wrt this argument using only known quantities as inputs.
#
# Ignoring issues of unbroadcasting (which we'll cover later), we could write the backward with respect to the first argument as:
#
# ```python
# def multiply_back(grad_out, out, a, b):
#     '''
#     Inputs:
#         grad_out = dL/d(out)
#         out = a * b
#
#     Returns:
#         dL/da
#     '''
#     return grad_out * b
# ```
#
# where `grad_out` is the gradient of the output of the function with respect to the loss (i.e. $\frac{dL}{dd}$), `out` is the output of the function (i.e. $d$), and `a` and `b` are our inputs.
#

# %% [markdown]
# ### Topological Ordering
#
# When we're actually doing backprop, how do we guarantee that we'll always know the value of our backwards functions' inputs? For instance, in the example above we couldn't have computed $\frac{dL}{da}$ without first knowing $\frac{dL}{dd}$.
#
# The answer is that we sort all our nodes using an algorithm called [topological sorting](https://en.wikipedia.org/wiki/Topological_sorting), and then do our computations in this order. After each computation, we store the gradients in our nodes for use in subsequent calculations.
#
# When described in terms of the diagram above, topological sort can be thought of as an ordering of nodes from right to left. Crucially, this sorting has the following property: if there is a directed path in the computational graph going from node `x` to node `y`, then `x` must follow `y` in the sorting.
#
# There are many ways of proving that a cycle-free directed graph contains a topological ordering. You can try and prove this for yourself, or click on the expander below to reveal the outline of a simple proof.
#

# %% [markdown]
# <details>
# <summary>Click to reveal proof</summary>
#
# We can prove by induction on the number of nodes $N$.
#
# If $N=1$, the problem is trivial.
#
# If $N>1$, then pick any node, and follow the arrows until you reach a node with no directed arrows going out of it. Such a node must exist, or else you would be following the arrows forever, and you'd eventually return to a node you previously visited, but this would be a cycle, which is a contradiction. Once you've found this "root node", you can put it first in your topological ordering, then remove it from the graph and apply the topological sort on the subgraph created by removing this node. By induction, your topological sorting algorithm on this smaller graph should return a valid ordering. If you append the root node to the start of this ordering, you have a topological ordering for the whole graph.
# </details>
#

# %% [markdown]
# A quick note on some potentially confusing terminology. We will refer to the "end node" as the **root node**, and the "starting nodes" as **leaf nodes**. For instance, in the diagram at the top of the section, the left nodes `a`, `b` and `c` are the leaf nodes, and `L` is the root node. This might seem odd given it makes the leaf nodes come before the root nodes, but the reason is as follows: *when we're doing the backpropagation algorithm, we start at `L` and work our way back through the graph*. So, by our notation, we start at the root node and work our way out to the leaves.
#
# Another important piece of terminology here is **parent node**. This means the same thing as it does in most other contexts - the parents of node `x` are all the nodes `y` with connections `y -> x` (so in the diagram, `L`'s parents are `d` and `e`).
#

# %% [markdown]
# <details>
# <summary>Question - can you think of a reason it might be important for a node to store a list of all of its parent nodes?</summary>
#
# During backprop, we're moving from right to left in the diagram. If a node doesn't store its parent, then there will be no way to get access to that parent node during backprop, so we can't propagate gradients to it.
# </details>
#

# %% [markdown]
# The very first node in our topological sort will be $L$, the root node.
#

# %% [markdown]
# ### Backpropagation
#
# After all this setup, the backpropagation mechanism becomes pretty straightforward. We sort the nodes topologically, then we iterate over them and call each backward function exactly once in order to accumulate the gradients at each node.
#
# It's important that the grads be accumulated instead of overwritten in a case like value $b$ which has two outgoing edges, since $\frac{dL}{db}$ will then be the sum of two terms. Since addition is commutative it doesn't matter whether we `backward()` the Mul or the Add that depend on $b$ first.
#
# During backpropagation, for each forward function in our computational graph we need to find the partial derivative of the output with respect to each of its inputs. Each partial is then multiplied by the gradient of the loss with respect to the forward functions output (`grad_out`) to find the gradient of the loss with respect to each input. We'll handle these calculations using backward functions.
#

# %% [markdown]
# ## Backward function of log
#
# First, we'll write the backward function for `x -> out = log(x)`. This should be a function which, when fed the values `x, out, grad_out = dL/d(out)` returns the value of `dL/dx` just from this particular computational path.
#

# %% [markdown]
# <img src="https://raw.githubusercontent.com/callummcdougall/Fundamentals/main/images/x_log_out.png" width="400">
#

# %% [markdown]
# Note - it might seem strange at first why we need `x` and `out` to be inputs, `out` can be calculated directly from `x`. The answer is that sometimes it is computationally cheaper to express the derivative in terms of `out` than in terms of `x`.
#

# %% [markdown]
# <details>
# <summary>Question - can you think of an example function where it would be computationally cheaper to use 'out' than to use 'x'?</summary>
#
# The most obvious answer is the exponential function, `out = e ^ x`. Here, the gradient `d(out)/dx` is equal to `out`. We'll see this when we implement a backward version of `torch.exp` later today.
# </details>
#

# %% [markdown]
# ### Exercise - implement `log_back`
#
# ```yaml
# Difficulty: 🔴🔴⚪⚪⚪
# Importance: 🔵🔵🔵⚪⚪
#
# You should spend up to 5-10 minutes on this exercise.
# ```
#

# %% [markdown]
# You should fill in this function below. Don't worry about division by zero or other edge cases - the goal here is just to see how the pieces of the system fit together.
#
# *This should just be a short, one-line function.*
#


# %%
def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backwards function for f(x) = log(x)

    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x).
    x: the input of np.log.

    Return: gradient of the given loss wrt x
    """
    return grad_out / x


# tests.test_log_back(log_back)

# %% [markdown]
#
#
# <details>
# <summary>Help - I get <code>ImportError: numpy.core.multiarray failed to import</code></summary>
#
# This is an annoying colab-related error which I haven't been able to find a satisfying fix for. The setup code at the top of this notebook should have installed a version of numpy which works, although you'll have to click "Restart Runtime" (from the Runtime menu) to make this work. Make sure you don't re-run the cell with `pip install`s.
#
# If this still doesn't work, please reach out to me at `callum@arena.education`.
# </details>
#

# %% [markdown]
# <details>
# <summary>Help - I'm not sure what the output of this backward function for log should be.</summary>
#
# By the chain rule, we have:
#
# $$
# \frac{dL}{dx} = \frac{dL}{d(\text{out})} \cdot \frac{d(\text{out})}{dx} = \frac{dL}{d(\text{out})} \cdot \frac{d(\log{x})}{dx} = \frac{dL}{d(\text{out})} \cdot \frac{1}{x}
# $$
#
# ---
#
# (Note - technically, $\frac{d(\text{out})}{dx}$ is a tensor containing the derivatives of each element of $\text{out}$ with respect to each element of $x$, and we should matrix multiply when we use the chain rule. However, since $\text{out} = \log x$ is an elementwise function of $x$, our application of the chain rule here will also be an elementwise multiplication: $\frac{dL}{dx_{ij}} = \frac{dL}{d(\text{out}_{ij})} \cdot \frac{d(\text{out}_{ij})}{dx_{ij}}$. When we get to things like matrix multiplication later, we'll have to be a bit more careful!)
# </details>
#

# %% [markdown]
# ## Backward functions of two tensors
#
# Now we'll implement backward functions for multiple tensors. To do so, we first need to understand broadcasting.
#

# %% [markdown]
# ### Broadcasting Rules
#
# Both NumPy and PyTorch have the same rules for broadcasting. The shape of the arrays being operated on is compared element-wise, starting from the rightmost dimension and working left. Two dimensions are compatible when
#
# * they are equal, or
# * one of them is 1 (in which case the array is repeated along this dimension to fit into the other one).
#
# Two arrays with a different number of dimensions can be operated on, provided the one with fewer dimensions is compatible with the rightmost elements of the one with more dimensions. Another way to picture this is that NumPy appends dimensions of size 1 to the start of the smaller-dimensional array until they both have the same dimensionality, and then their sizes are checked for compatibility.
#
# As a warm-up exercise, below are some examples of broadcasting. Can you figure out which are valid, and which will raise errors?
#

# %% [markdown]
# ```python
# x = np.ones((3, 1, 5))
# y = np.ones((1, 4, 5))
#
# z = x + y
# ```
#
# <details>
# <summary>Answer</summary>
#
# This is valid, because the 0th dimension of `y` and the 1st dimension of `x` can both be copied so that `x` and `y` have the same shape: `(3, 4, 5)`. The resulting array `z` will also have shape `(3, 4, 5)`.
# </details>
#

# %% [markdown]
# ```python
# x = np.ones((8, 2, 6))
# y = np.ones((8, 2))
#
# z = x + y
# ```
#
# <details>
# <summary>Answer</summary>
#
# This is not valid. We first need to expand `y` by appending a dimension to the front, and the last two dimensions of `x` are `(2, 6)`, which won't broadcast with `y`'s `(8, 2)`.
# </details>
#

# %% [markdown]
# ```python
# x = np.ones((8, 2, 6))
# y = np.ones((2, 6))
#
# z = x + y
# ```
#
# <details>
# <summary>Answer</summary>
#
# This is valid. Once NumPy expands `y` by appending a single dimension to the front, it can then be broadcast with `x`.
# </details>
#

# %% [markdown]
# ### Why do we need broadcasting for backprop?
#

# %% [markdown]
# Often, a tensor $x$ gets broadcasted to produce another tensor $x_{broadcasted}$, when being used to create $out$. It might be easy to define the derivative wrt $out$ and $x_{broadcasted}$, but we need to know how to go from this to calculating the derivative wrt $x$.
#
# To take an example:
#
# ```python
# x = t.ones(4,)
# y = t.ones(3, 4)
# out = x + y # = x_broadcasted + y
# L = out[0, 0] + out[1, 1] + out[2, 1]
# ```
#

# %% [markdown]
# <img src="https://raw.githubusercontent.com/callummcdougall/Fundamentals/main/images/xy_add_out.png" width="400">
#

# %% [markdown]
# <img src="https://raw.githubusercontent.com/callummcdougall/Fundamentals/main/images/broadcast-2.png" width="400">
#

# %% [markdown]
# In this case, we have:
#
# $$
# \frac{dL}{d(out)} = \frac{dL}{dx_{broadcasted}} = \begin{bmatrix}
# 1 & 0 & 0 & 0\\
# 0 & 1 & 0 & 0\\
# 0 & 1 & 0 & 0
# \end{bmatrix}
# $$
#
# How do we get from this to $\frac{dL}{dx}$? Well, we can write $L$ as a function of $x$ (ignoring $y$ for now):
#
# $$
# \begin{aligned}
# L &= x_{broadcasted}[0, 0] + x_{broadcasted}[1, 1] + x_{broadcasted}[2, 1] \\
# &= x[0] + x[1] + x[1] \\
# &= x[0] + 2x[1]
# \end{aligned}
# $$
#
# meaning the derivative with respect to $x$ is:
#
# $$
# \frac{dL}{dx} = \begin{bmatrix}
# 1 & 2 & 0 & 0
# \end{bmatrix}
# $$
#
# Note how we got this by taking $\frac{dL}{dx_{broadcasted}}$, and summing it over the dimension along which $x$ was broadcasted. This leads to our general rule for handling broadcasted operations:
#

# %% [markdown]
# > ##### Summary
# >
# > If we know $\frac{dL}{d(out)}$, and want to know $\frac{dL}{dx}$ (where $x$ was broadcasted to produce $out$) then there are two steps:
# >
# > 1. Compute $\frac{dL}{dx_{broadcasted}}$ in the standard way, i.e. using one of your backward functions (no broadcasting involved here).
# > 2. ***Unbroadcast*** $\frac{dL}{dx_{broadcasted}}$, by summing it over the dimensions along which $x$ was broadcasted.
#

# %% [markdown]
# We used the term "unbroadcast" because the way that our tensor's shape changes will be the reverse of how it changed during broadcasting. If `x` was broadcasted from `(4,) -> (3, 4)`, then unbroadcasting will have to take a tensor of shape `(3, 4)` and sum over it to return a tensor of shape `(4,)`. Similarly, if `x` was broadcasted from `(1, 4) -> (3, 4)`, then we need to sum over the zeroth dimension, but leave it as a 2D tensor (with zeroth dimension of size 1).
#

# %% [markdown]
# ### Exercise - implement `unbroadcast`
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵⚪⚪⚪
#
# You should spend up to 15-20 minutes on this exercise.
#
# This can be finnicky to implement, so you should be willing to read the solution and move on if you get stuck.
# ```
#
# Below, you should implement this function. `broadcasted` is the array you want to sum over, and `original` is the array with the shape you want to return. Your function should:
#
# * Compare the shapes of `broadcasted` and `original`, and deduce (using broadcasting rules) how `original` was broadcasted to get `broadcasted`.
# * Sum over the dimensions of `broadcasted` that were added by broadcasting, and return a tensor of shape `original.shape`.
#
# Hint - the `.sum()` method (for NumPy arrays) takes arguments `axis` and `keepdims`. The `axis` argument is an int or list of ints to sum over, and `keepdims` is a boolean that determines whether you want to remove the dims you're summing over (if `False`) or leave them as dims of size 1 (if `True`). You'll need to use both arguments when implementing this function.
#


# %%
def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    """
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    """
    # find the dimensions that were broadcasted
    # original has to have 1 in some dimension to be broadcasted
    # print(broadcasted.shape, original.shape)
    nd_original = len(original.shape)
    nd_broadcasted = len(broadcasted.shape)
    # first find if dimensions were added to the original tensor
    added_dims = tuple(range(nd_broadcasted - nd_original))
    # sum over these dimensions
    unb = broadcasted.sum(axis=added_dims, keepdims=False)

    broadcasted_dims = []
    for i in range(len(unb.shape)):
        # this dim was 1 and the broadcasted dim is not 1
        orig_dim = original.shape[i]
        broadcasted_dim = unb.shape[i]

        if orig_dim == 1 and broadcasted_dim != 1:
            # if original was 1, and broadcasted is not 1, means this dimension was broadcasted
            # add it to broadcasted_dims
            broadcasted_dims.append(i)

    # print(unb.shape, broadcasted_dims)
    unb = np.sum(unb, axis=tuple(broadcasted_dims), keepdims=True)

    return unb


# tests.test_unbroadcast(unbroadcast)

# %% [markdown]
# <details>
# <summary>Help - I'm confused about implementing unbroadcast!</summary>
#
# Recall that broadcasting `original -> broadcasted` has 2 steps:
#
# 1. Append dims of size 1 to the start of `original`, until it has the same number of dims as `broadcasted`.
# 2. Copy `original` along each dimension where it has size 1.
#
# Similarly, your `unbroadcast` function should have 2 steps:
#
# 1. Sum over the dimensions at the start of `broadcasted`, until the result has the same number of dims as `original`.
#     * Here you should use `keepdims=False`, because you're trying to reduce the dimensionality of `broadcasted`.
# 2. Sum over the dimensions of `broadcasted` wherever `original` has size 1.
#     * Here you should use `keepdims=True`, because you want to leave these dimensions having size 1 (so that the result has the same shape as `original`).
# </details>

# %% [markdown]
# ### Backward Function for Elementwise Multiply
#
# Functions that are differentiable with respect to more than one input tensor are straightforward given that we already know how to handle broadcasting.
#
# - We're going to have two backwards functions, one for each input argument.
# - If the input arguments were broadcasted together to create a larger output, the incoming `grad_out` will be of the larger common broadcasted shape and we need to make use of `unbroadcast` from earlier to match the shape to the appropriate input argument.
# - We'll want our backward function to work when one of the inputs is an float. We won't need to calculate the grad_in with respect to floats, so we only need to consider when y is an float for `multiply_back0` and when x is an float for `multiply_back1`.
#

# %% [markdown]
# ### Exercise - implement both `multiply_back` functions
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵⚪⚪⚪
#
# You should spend up to 10-15 minutes on these exercises.
# ```
#

# %% [markdown]
# Below, you should implement both `multiply_back0` and `multiply_back1`.
#
# You might be wondering why we need two different functions, rather than just having a single function to serve both purposes. This will become more important later on, once we deal with functions with more than one argument, which is not symmetric in its arguments. For instance, the derivative of $x / y$ wrt $x$ is not the same as the expression you get after differentiating this wrt $y$ then swapping the labels around.
#
# The first part of each function has been provided for you (this makes sure that both inputs are arrays).
#
# <details>
# <summary>Help - I'm not sure how to use the <code>unbroadcast</code> function.</summary>
#
# First, do the calculation assuming no broadcasting. Then, use `unbroadcast` to make sure the result has the same shape as the array you're trying to calculate the derivative with respect to.
# </details>
#


# %%
def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr | float) -> Arr:
    """Backwards function for x * y wrt argument 0 aka x."""
    if not isinstance(y, Arr):
        y = np.array(y)

    # the deriviative is grad_out * y
    # we know that y could have been broadcasted to form something that has the same shape as out
    return unbroadcast(grad_out * y, x)


def multiply_back1(grad_out: Arr, out: Arr, x: Arr | float, y: Arr) -> Arr:
    """Backwards function for x * y wrt argument 1 aka y."""
    if not isinstance(x, Arr):
        x = np.array(x)

    return unbroadcast(grad_out * x, y)


# tests.test_multiply_back(multiply_back0, multiply_back1)
# tests.test_multiply_back_float(multiply_back0, multiply_back1)

# %% [markdown]
# Now we'll use our backward functions to do backpropagation manually, for the following computational graph:
#
# <img src="https://raw.githubusercontent.com/callummcdougall/Fundamentals/main/images/abcdefg.png" width=550>
#

# %% [markdown]
# ### Exercise - implement `forward_and_back`
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to 15-20 minutes on these exercises.
#
# This function is very useful for getting a hands-on sense of the backpropagation algorithm.
# ```
#
# Below, you should implement the `forward_and_back` function. This is an opportunity for you to practice using the backward functions you've written so far, and should hopefully give you a better sense of how the full backprop function will eventually work.
#
# Note - we're assuming all arrays in this graph have size 1, i.e. they're just scalars.
#


# %%
def forward_and_back(a: Arr, b: Arr, c: Arr) -> tuple[Arr, Arr, Arr]:
    """
    Calculates the output of the computational graph above (g), then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    """
    d = a * b
    e = np.log(c)
    f = d * e
    g = np.log(f)
    dg_dg = 1
    dg_df = log_back(dg_dg, g, f)
    dg_dd = multiply_back0(dg_df, f, d, e)
    dg_de = multiply_back1(dg_df, f, d, e)
    dg_dc = log_back(dg_de, e, c)
    dg_da = multiply_back0(dg_dd, d, a, b)
    dg_db = multiply_back1(dg_dd, d, a, b)

    return dg_da, dg_db, dg_dc


# tests.test_forward_and_back(forward_and_back)

# %% [markdown]
# <details>
# <summary>Help - I'm not sure what my first 'grad_out' argument should be!</summary>
#
# `grad_out` is $\frac{dL}{d(\text{out})}$, where $L$ is the node at the end of your graph and $\text{out}$ is the output of the function you're backpropagating through. For your first function (which is `log : f -> g`), the output is `g`, so your `grad_out` should be $\frac{dg}{dg} = 1$. (You should make this an array of size 1, since `g` is a scalar.)
#
# The output of this function is $\frac{dg}{df}$, which you can use as the `grad_out` argument in subsequent backward funcs.
# </details>

# %% [markdown]
# In the next section, you'll build up to full automation of this backpropagation process, in a way that's similar to PyTorch's `autograd`.
#

# %% [markdown]
# # 2️⃣ Autograd
#

# %% [markdown]
# > ## Learning Objectives
# >
# > * Perform a topological sort of a computational graph (and understand why this is important).
# > * Implement a the `backprop` function, to calculate and store gradients for all tensors in a computational graph.
#

# %% [markdown]
# Now, rather than figuring out which backward functions to call, in what order, and what their inputs should be, we'll write code that takes care of that for us. We'll implement this with a few major components:
# - Tensor
# - Recipe
# - wrap_forward_fn
#

# %% [markdown]
# ## Wrapping Arrays (Tensor)
#
# We're going to wrap each array with a wrapper object from our library which we'll call `Tensor` because it's going to behave similarly to a `torch.Tensor`.
#
# Each Tensor that is created by one of our forward functions will have a `Recipe`, which tracks the extra information need to run backpropagation.
#
# `wrap_forward_fn` will take a forward function and return a new forward function that does the same thing while recording the info we need to do backprop in the `Recipe`.
#

# %% [markdown]
# ## Recipe
#
# Let's start by taking a look at `Recipe`.
#
# `@dataclass` is a handy class decorator that sets up an `__init__` function for the class that takes the provided attributes as arguments and sets them as you'd expect.
#
# The class `Recipe` is designed to track the forward functions in our computational graph, so that gradients can be calculated during backprop. Each tensor created by a forward function has its own `Recipe`. We're naming it this because it is a set of instructions that tell us which ingredients went into making our tensor: what the function was, and what tensors were used as input to the function to produce this one as output.
#


# %%
@dataclass(frozen=True)
class Recipe:
    """Extra information necessary to run backpropagation. You don't need to modify this."""

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll create for it later on."

    args: tuple
    "The input arguments passed to func."
    "For instance, if func was np.sum then args would be a length-1 tuple containing the tensor to be summed."

    kwargs: Dict[str, Any]
    "Keyword arguments passed to func."
    "For instance, if func was np.sum then kwargs might contain 'dim' and 'keepdims'."

    parents: Dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position, in order to be able to pass gradients back along the computational graph."


# %% [markdown]
# Note that `args` just stores the values of the underlying arrays, but `parents` stores the actual tensors. This is because they serve two different purposes: `args` is required for computing the value of gradients during backpropagation, and `parents` is required to infer the structure of the computational graph (i.e. which tensors were used to produce which other tensors).
#
# Here are some examples, to build intuition for what the four fields of `Recipe` are, and why we need all four of them to fully describe a tensor in our graph and how it was created:
#
# <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/recipe-fixed-2.png" width="800">
#

# %% [markdown]
# ## Registering backwards functions
#
# The `Recipe` takes care of tracking the forward functions in our computational graph, but we still need a way to find the backward function corresponding to a given forward function when we do backprop (or possibly the set of backward functions, if the forward function takes more than one argument).
#

# %% [markdown]
# ### Exercise - implement `BackwardFuncLookup`
#
# ```yaml
# Difficulty: 🔴🔴⚪⚪⚪
# Importance: 🔵🔵🔵⚪⚪
#
# You should spend up to 10-15 minutes on these exercises.
#
# These exercises should be very short, once you understand what is being asked.
# ```
#

# %% [markdown]
# We will define a class `BackwardFuncLookup` in order to find the backward function for a given forward function. Details of the implementation are left up to you.
#
# The implementation today can be done very simply. We won't support backprop wrt keyword arguments and will raise an exception if the user tries to pass a Tensor by keyword. You can remove this limitation later if you have time.
#
# We do need to support functions with multiple positional arguments like multiplication so we'll also provide the positional argument index when setting and getting back_fns.
#
# If you're confused as to what this question is asking you to implement, you can look at the code below the class definition (which shows how a class instance can should be used to store and access backward functions).
#


# %%
class BackwardFuncLookup:
    def __init__(self) -> None:
        self.back_fns: defaultdict[Callable, dict[int, Callable]] = defaultdict(dict)

    def add_back_func(
        self, forward_fn: Callable, arg_position: int, back_fn: Callable
    ) -> None:
        self.back_fns[forward_fn][arg_position] = back_fn

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        return self.back_fns[forward_fn][arg_position]


BACK_FUNCS = BackwardFuncLookup()
BACK_FUNCS.add_back_func(np.log, 0, log_back)
BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)

# assert BACK_FUNCS.get_back_func(np.log, 0) == log_back
# assert BACK_FUNCS.get_back_func(np.multiply, 0) == multiply_back0
# assert BACK_FUNCS.get_back_func(np.multiply, 1) == multiply_back1

# print("Tests passed - BackwardFuncLookup class is working as expected!")

# %% [markdown]
# <details>
# <summary>Example implementation</summary>
#
# This implementation uses the useful [`collections.defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict) item.
#
# Also, note the use of type annotations. This allows you to auto-fill appropriate methods when you work with these objects.
#
# ```python
# class BackwardFuncLookup:
#     def __init__(self) -> None:
#         self.back_funcs: defaultdict[Callable, dict[int, Callable]] = defaultdict(dict)
#
#     def add_back_func(self, forward_fn: Callable, arg_position: int, back_fn: Callable) -> None:
#         self.back_funcs[forward_fn][arg_position] = back_fn
#
#     def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
#         return self.back_funcs[forward_fn][arg_position]
# ```
# </details>
#

# %% [markdown]
# ## Tensors
#
# Our Tensor object has these fields:
# - An `array` field of type `np.ndarray`.
# - A `requires_grad` field of type `bool`.
# - A `grad` field of the same size and type as the value.
# - A `recipe` field, as we've already seen.
#

# %% [markdown]
# ### requires_grad
#
# The meaning of `requires_grad` is that when doing operations using this tensor, the recipe will be stored and it and any descendents will be included in the computational graph.
#
# Note that `requires_grad` does not necessarily mean that we will save the accumulated gradients to this tensor's `.grad` parameter when doing backprop: we will follow pytorch's implementation of backprop and only save gradients to leaf tensors (see `Tensor.is_leaf`, below).
#
# ---
#
# There is a lot of repetitive boilerplate involved which we have done for you. You don't need to modify anything in this class: the methods here will delegate to functions that you will implement throughout the day. You should read the code for the `Tensor` class up to `__init__`, and make sure you understand it. Most of the methods beyond this are just replicating the basic functionality of PyTorch tensors.
#

# %%
Arr = np.ndarray


class Tensor:
    """
    A drop-in replacement for torch.Tensor supporting a subset of features.
    """

    array: Arr
    "The underlying array. Can be shared between multiple Tensors."
    requires_grad: bool
    "If True, calling functions or methods on this tensor will track relevant data for backprop."
    grad: "Tensor | None"
    "Backpropagation will accumulate gradients into this field."
    recipe: Recipe | None
    "Extra information necessary to run backpropagation."

    def __init__(self, array: Arr | list, requires_grad=False):
        self.array = array if isinstance(array, Arr) else np.array(array)
        if self.array.dtype == np.float64:
            self.array = self.array.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.recipe = None
        "If not None, this tensor's array was created via recipe.func(*recipe.args, **recipe.kwargs)."

    def __neg__(self) -> "Tensor":
        return negative(self)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return subtract(self, other)

    def __rsub__(self, other) -> "Tensor":
        return subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __eq__(self, other):
        return eq(self, other)

    def __repr__(self) -> str:
        return f"Tensor({repr(self.array)}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if self.array.ndim == 0:
            raise TypeError
        return self.array.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, index) -> "Tensor":
        return getitem(self, index)

    def add_(self, other: "Tensor", alpha: float = 1.0) -> "Tensor":
        add_(self, other, alpha=alpha)
        return self

    @property
    def T(self) -> "Tensor":
        return permute(self, axes=(-1, -2))

    def item(self):
        return self.array.item()

    def sum(self, dim=None, keepdim=False) -> "Tensor":
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self) -> "Tensor":
        return log(self)

    def exp(self) -> "Tensor":
        return exp(self)

    def reshape(self, new_shape) -> "Tensor":
        return reshape(self, new_shape)

    def expand(self, new_shape) -> "Tensor":
        return expand(self, new_shape)

    def permute(self, dims) -> "Tensor":
        return permute(self, dims)

    def maximum(self, other) -> "Tensor":
        return maximum(self, other)

    def relu(self) -> "Tensor":
        return relu(self)

    def argmax(self, dim=None, keepdim=False) -> "Tensor":
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low: float, high: float) -> "Tensor":
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad: Union[Arr, "Tensor", None] = None) -> None:
        if isinstance(end_grad, Arr):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        """Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html"""
        if self.requires_grad and self.recipe and self.recipe.parents:
            return False
        return True

    def __bool__(self):
        if np.array(self.shape).prod() != 1:
            raise RuntimeError(
                "bool value of Tensor with more than one value is ambiguous"
            )
        return bool(self.item())


def empty(*shape: int) -> Tensor:
    """Like torch.empty."""
    return Tensor(np.empty(shape))


def zeros(*shape: int) -> Tensor:
    """Like torch.zeros."""
    return Tensor(np.zeros(shape))


def arange(start: int, end: int, step=1) -> Tensor:
    """Like torch.arange(start, end)."""
    return Tensor(np.arange(start, end, step=step))


def tensor(array: Arr, requires_grad=False) -> Tensor:
    """Like torch.tensor."""
    return Tensor(array, requires_grad=requires_grad)


# %% [markdown]
# ## Forward Pass: Building the Computational Graph
#
# Let's start with a simple case: our `log` function. `log_forward` is a wrapper, which should implement the functionality of `np.log` but work with tensors rather than arrays.
#

# %% [markdown]
# ### Exercise - implement `log_forward`
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵🔵⚪⚪
#
# You should spend up to 15-20 minutes on this exercise.
# ```
#
# Our `log` function must do the following:
#
# - Call `np.log` on the input *array* (i.e. the array attribute of the tensor).
# - Create a new `Tensor` containing the output.
# - If grad tracking is enabled globally AND (the input requires grad, OR has a recipe), then the output requires grad and we fill out the recipe of our output, as a `Recipe` object.
#
# Later we'll redo this in a generic and reusable way, but for now just get it working.
#
# Note - in these exercises, we've used the global variable `grad_tracking_enabled` to indicate whether gradient tracking is enabled. Often when you're working with a model and you don't need to train it (or perform backward passes for any other reason) it's useful to disable gradient tracking. In PyTorch, this is done with `t.set_grad_enabled(False)`.
#


# %%
def log_forward(x: Tensor) -> Tensor:
    """Performs np.log on a Tensor object."""
    out = np.log(x.array)
    recipe = None

    if grad_tracking_enabled and (x.requires_grad or x.recipe):
        recipe = Recipe(np.log, (x.array,), kwargs={}, parents={0: x})

    t = Tensor(out, requires_grad=x.requires_grad and grad_tracking_enabled)
    t.recipe = recipe

    return t


log = log_forward
# tests.test_log(Tensor, log_forward)
# tests.test_log_no_grad(Tensor, log_forward)
# a = Tensor([1], requires_grad=True)
# grad_tracking_enabled = False
# b = log_forward(a)
# grad_tracking_enabled = True
# assert not b.requires_grad, "should not require grad if grad tracking globally disabled"
# assert b.recipe is None, "should not create recipe if grad tracking globally disabled"

# %% [markdown]
# <details>
# <summary>Help - I need more hints on how to implement this function.</summary>
#
# You need to define a tensor `out` by feeding it the underlying data (log of `x.array`) and the `requires_grad` flag.
#
# Then, if `requires_grad` is true, you should also create a recipe object and store it in `out`. You can look at the diagrams above to see what the recipe should look like (it will be even simpler than the ones pictured, because there's only one parent, one arg, and no kwargs).
# </details>
#

# %% [markdown]
# Now let's do the same for multiply, to see how to handle functions with multiple arguments.
#

# %% [markdown]
# ### Exercise - implement `multiply_forward`
#
# ```yaml
# Difficulty: 🔴🔴🔴🔴⚪
# Importance: 🔵🔵🔵⚪⚪
#
# You should spend up to 15-20 minutes on this exercise.
# ```
#
# There are a few differences between this and log:
#
# - The actual function to be called is different
# - We need more than one argument in `args` and `parents`, when defining `Recipe`
# - `requires_grad` should be true if `grad_tracking_enabled=True`, and ANY of the input tensors require grad
# - One of the inputs may be an int, so you'll need to deal with this case before calculating `out`
#
# If you're confused, you can scroll up to the diagram at the top of the page (which tells you how to construct the recipe for functions like multiply or add when they are both arrays, or when one is an array and the other is a scalar).
#


# %%
def multiply_forward(a: Tensor | int, b: Tensor | int) -> Tensor:
    """Performs np.multiply on a Tensor object."""
    assert isinstance(a, Tensor) or isinstance(b, Tensor)

    requires_grad = False
    if not isinstance(a, Tensor):
        out = np.multiply(a, b.array)
        parents = {1: b}
        args = (a, b.array)
        requires_grad = grad_tracking_enabled and b.requires_grad
    elif not isinstance(b, Tensor):
        out = np.multiply(a.array, b)
        parents = {0: a}
        args = (a.array, b)
        requires_grad = grad_tracking_enabled and a.requires_grad
    else:
        out = np.multiply(a.array, b.array)
        parents = {0: a, 1: b}
        args = (a.array, b.array)
        requires_grad = grad_tracking_enabled and (a.requires_grad or b.requires_grad)

    recipe = None
    if grad_tracking_enabled and requires_grad:
        recipe = Recipe(np.multiply, args=args, kwargs={}, parents=parents)

    t = Tensor(out, requires_grad=requires_grad)
    t.recipe = recipe

    return t


multiply = multiply_forward
# tests.test_multiply(Tensor, multiply_forward)
# tests.test_multiply_no_grad(Tensor, multiply_forward)
# tests.test_multiply_float(Tensor, multiply_forward)
# a = Tensor([2], requires_grad=True)
# b = Tensor([3], requires_grad=True)
# grad_tracking_enabled = False
# b = multiply_forward(a, b)
# grad_tracking_enabled = True
# assert not b.requires_grad, "should not require grad if grad tracking globally disabled"
# assert b.recipe is None, "should not create recipe if grad tracking globally disabled"

# %% [markdown]
# <details>
# <summary>Help - I get "AttributeError: 'int' object has no attribute 'array'".</summary>
#
# Remember that your multiply function should also accept integers. You need to separately deal with the cases where `a` and `b` are integers or Tensors.
# </details>
#
# <details>
# <summary>Help - I get "AssertionError: assert len(c.recipe.parents) == 1 and c.recipe.parents[0] is a" in the "test_multiply_float" test.</summary>
#
# This is probably because you've stored the inputs to `multiply` as integers when one of the is an integer. Remember, `parents` should just be a list of the **Tensors** that were inputs to `multiply`, so you shouldn't add ints.
# </details>
#

# %% [markdown]
# ## Forward Pass - Generic Version
#
# All our forward functions are going to look extremely similar to `log_forward` and `multiply_forward`.
# Implement the higher order function `wrap_forward_fn` that takes a `Arr -> Arr` function and returns a `Tensor -> Tensor` function. In other words, `wrap_forward_fn(np.multiply)` should evaluate to a callable that does the same thing as your `multiply_forward` (and same for `np.log`).
#

# %% [markdown]
# ### Exercise - implement `wrap_forward_fn`
#
# ```yaml
# Difficulty: 🔴🔴🔴🔴⚪
# Importance: 🔵🔵🔵⚪⚪
#
# You should spend up to 20-25 minutes on this exercise.
#
# This exercise is conceptually important, but might be a bit difficult if it isn't clear what the question is asking for.
# ```
#


# %%
def wrap_forward_fn(numpy_func: Callable, is_differentiable=True) -> Callable:
    """
    numpy_func: Callable
        takes any number of positional arguments, some of which may be NumPy arrays, and
        any number of keyword arguments which we aren't allowing to be NumPy arrays at
        present. It returns a single NumPy array.

    is_differentiable:
        if True, numpy_func is differentiable with respect to some input argument, so we
        may need to track information in a Recipe. If False, we definitely don't need to
        track information.

    Return: Callable
        It has the same signature as numpy_func, except wherever there was a NumPy array,
        this has a Tensor instead.
    """

    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        parents = {i: arg for i, arg in enumerate(args) if isinstance(arg, Tensor)}
        requires_grad = (
            grad_tracking_enabled
            and is_differentiable
            and any(
                [
                    arg.requires_grad if isinstance(arg, Tensor) else False
                    for arg in args
                ]
            )
        )
        recipe = None
        numpy_fn_args = [arg.array if isinstance(arg, Tensor) else arg for arg in args]

        out = numpy_func(*numpy_fn_args, **kwargs)
        if requires_grad:
            recipe = Recipe(
                numpy_func, args=tuple(numpy_fn_args), kwargs=kwargs, parents=parents
            )

        t = Tensor(out, requires_grad=requires_grad)
        t.recipe = recipe
        return t

    return tensor_func


def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    # need to be careful with sum, because kwargs have different names in torch and numpy
    return np.sum(x, axis=dim, keepdims=keepdim)


log = wrap_forward_fn(np.log)
multiply = wrap_forward_fn(np.multiply)
eq = wrap_forward_fn(np.equal, is_differentiable=False)
sum = wrap_forward_fn(_sum)

# tests.test_log(Tensor, log)
# tests.test_log_no_grad(Tensor, log)
# tests.test_multiply(Tensor, multiply)
# tests.test_multiply_no_grad(Tensor, multiply)
# tests.test_multiply_float(Tensor, multiply)
# tests.test_sum(Tensor)

# %% [markdown]
# <details>
# <summary>Help - I'm not sure where to start.</summary>
#
# Start with the code from `multiply_forward`. The way this function was structured (i.e. the five comments in the solution) should also give you a good template for how to structure your `tensor_func` function.
# </details>
#
# <details>
# <summary>Help - I'm getting <code>NameError: name 'getitem' is not defined</code>.</summary>
#
# This is probably because you're calling `numpy_func` on the args themselves. Recall that `args` will be a list of `Tensor` objects, and that you should call `numpy_func` on the underlying arrays.
# </details>
#
# <details>
# <summary>Help - I'm getting an AssertionError on <code>assert c.requires_grad == True</code> (or something similar).</summary>
#
# This is probably because you're not defining `requires_grad` correctly. Remember that the output of a forward function should have `requires_grad = True` if and only if all of the following hold:
#
# * Grad tracking is enabled
# * The function is differentiable
# * **Any** of the inputs are tensors with `requires_grad = True`
# </details>
#
# <details>
# <summary>Help - my function passes all tests up to <code>test_sum</code>, but then fails here.</summary>
#
# `test_sum`, unlike the previous tests, wraps a function that uses keyword arguments. So if you're failing here, it's probably because you didn't use `kwargs` correctly.
#
# `kwargs` should be used in two ways: once when actually calling the `numpy_func`, and once when defining the `Recipe` object for the output tensor.
# </details>
#

# %% [markdown]
# Note - none of these functions involve keyword args, so the tests won't detect if you're handling kwargs incorrectly (or even failing to use them at all). If your code fails in later exercises, you might want to come back here and check that you're using the kwargs correctly. Alternatively, once you pass the tests, you can compare your code to the solutions and see how they handle kwargs.
#

# %% [markdown]
# ## Backpropagation
#
# Now all the pieces are in place to implement backpropagation. We need to:
# - Loop over the nodes from right to left. At each node:
#     - Call the backward function to transform the grad wrt output to the grad wrt input.
#     - If the node is a leaf, write the grad to the grad field.
#     - Otherwise, accumulate the grad into temporary storage.
#

# %% [markdown]
# ### Topological Sort
#
# As part of backprop, we need to sort the nodes of our graph so we can traverse the graph in the appropriate order.
#

# %% [markdown]
# ### Exercise - implement `topological_sort`
#
# ```yaml
# Difficulty: 🔴🔴🔴🔴⚪
# Importance: 🔵⚪⚪⚪⚪
#
# You should spend up to 20-25 minutes on this exercise.
#
# Note, it's completely fine to skip this problem if you're not very interested in it. This is more of a fun LeetCode-style puzzle, and writing a solution for this isn't crucial for the overall experience of these exercises.
# ```
#

# %% [markdown]
# Write a general function `topological_sort` that return a list of node's children in topological order (beginning with the furthest descendants, ending with the starting node) using [depth-first search](https://en.wikipedia.org/wiki/Topological_sorting).
#
# We've given you a `Node` class, with a `children` attribute, and a `get_children` function. You shouldn't change any of these, and your `topological_sort` function should use `get_children` to access a node's children rather than calling `node.children` directly. In subsequent exercises, we'll replace the `Node` class with the `Tensor` class (and using a different `get_children` function), so this will ensure your code still works for this new case.
#
# If you're stuck, try looking at the pseudocode from some of [these examples](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm).
#


# %%
class Node:
    def __init__(self, *children):
        self.children = list(children)


def get_children(node: Node) -> list[Node]:
    return node.children


def topological_sort(node: Node, get_children: Callable) -> list[Node]:
    """
    Return a list of node's descendants in reverse topological order from future to past (i.e. `node` should be last).

    Should raise an error if the graph with `node` as root is not in fact acyclic.
    """
    sorted_nodes = []
    # we HAVE to do use a set for checking in this method becuase
    # if we have a list, and check `node in sorted_nodes`
    # by default a list will be checked for equality (the __eq__ method).
    # but it is not defined for tensors yet in this file hence this fails
    # while in a set, it uses __hash__ so still works here.
    # stolen fromthe solution which says "this is faster", which is also true. O(1) vs O(n).
    sorted_nodes_set = set()
    cycle = set()

    def visit(n):
        # "cycle" is keeping a track of nodes we visit when we're dfs-ing a specific node
        # if, while dfs-ing we visit a node twice, means a child's children was the parent node
        # since it exists in cycle, means we haven't yet run cycle.remove() on the parent
        # means we're still inside the for loop of get_children.
        # fucked. error out.
        if n in cycle:
            raise Exception("Cyclic graph detected")

        cycle.add(n)

        # but this is okay. because a leaf node can be a child of multiple parents. we only need to visit it once.
        if n in sorted_nodes_set:
            return

        for c in get_children(n):
            visit(c)

        # remove this from the cuycle
        cycle.remove(n)
        sorted_nodes.append(n)
        sorted_nodes_set.add(n)

    visit(node)
    return sorted_nodes


# tests.test_topological_sort_linked_list(topological_sort)
# tests.test_topological_sort_branching(topological_sort)
# tests.test_topological_sort_rejoining(topological_sort)
# tests.test_topological_sort_cyclic(topological_sort)

# %% [markdown]
# <details>
# <summary>Help - my function is hanging without returning any values.</summary>
#
# This is probably because it's going around in cycles when fed a cyclic graph. You should add a way of raising an error if your function detects that the graph isn't cyclic. One way to do this is to create a set `temp`, which stores the nodes you've visited on a particular excursion into the graph, then you can raise an error if you come across an already visited node.
# </details>
#
# <details>
# <summary>Help - I'm completely stuck on how to implement this, and would like the template for some code.</summary>
#
# Here is the template for a depth-first search implementation:
#
# ```python
# def topological_sort(node: Node, get_children: Callable) -> list[Node]:
#
#     result: list[Node] = [] # stores the list of nodes to be returned (in reverse topological order)
#     perm: set[Node] = set() # same as `result`, but as a set (faster to check for membership)
#     temp: set[Node] = set() # keeps track of previously visited nodes (to detect cyclicity)
#
#     def visit(cur: Node):
#         '''
#         Recursive function which visits all the children of the current node, and appends them all
#         to `result` in the order they were found.
#         '''
#         pass
#
#     visit(node)
#     return result
# ```
# </details>
#

# %% [markdown]
# Now, you should write the function `sorted_computational_graph`. This should be a short function (the main part of it is calling `topological_sort`), but there are a few things to keep in mind:
#
# * You'll need a different `get_children` function for when you call `topological_sort`. This should actually return the **parents** of the tensor in question (sorry for the confusing terminology!).
# * You should return the tensors in the order needed for backprop, in other words the `tensor` argument should be the first one in your list.
#

# %% [markdown]
# <img src="https://github.com/callummcdougall/Fundamentals/blob/main/images/abcdefg.png?raw=true" width=500>
#


# %%
def sorted_computational_graph(tensor: Tensor) -> list[Tensor]:
    """
    For a given tensor, return a list of Tensors that make up the nodes of the given Tensor's computational graph,
    in reverse topological order (i.e. `tensor` should be first).
    """

    def get_children(t):
        if not t.recipe:
            return []
        return list(t.recipe.parents.values())

    return list(reversed(topological_sort(tensor, get_children)))


# a = Tensor([1], requires_grad=True)
# b = Tensor([2], requires_grad=True)
# c = Tensor([3], requires_grad=True)
# d = a * b
# e = c.log()
# f = d * e
# g = f.log()
# name_lookup = {a: "a", b: "b", c: "c", d: "d", e: "e", f: "f", g: "g"}

# print([name_lookup[t] for t in sorted_computational_graph(g)])

# # %%
# a = Tensor([1], requires_grad=True)
# # a2 = Tensor([1], requires_grad=True)
# b = a * 2
# c = a * 1
# d = b * c
# name_lookup = {a: "a", b: "b", c: "c", d: "d"}

# print([name_lookup[t] for t in sorted_computational_graph(d)])

# %% [markdown]
# Compare your output with the computational graph. You should never be printing `x` before `y` if there is an edge `x --> ... --> y` (this should result in approximately reverse alphabetical order).
#

# %% [markdown]
# ### The `backward` method
#
# Now we're really ready for backprop!
#
# Recall that in the implementation of the class `Tensor`, we had:
#
# ```python
# class Tensor:
#
#     def backward(self, end_grad: "Arr | Tensor | None" = None) -> None:
#         if isinstance(end_grad, Arr):
#             end_grad = Tensor(end_grad)
#         return backprop(self, end_grad)
# ```
#
# In other words, for a tensor `out`, calling `out.backward()` is equivalent to `backprop(out)`.
#

# %% [markdown]
# ### End grad
#
# You might be wondering what role the `end_grad` argument in `backward` plays. We usually just call `out.backward()` when we're working with loss functions; why do we need another argument?
#
# The reason is that we've only ever called `tensor.backward()` on scalars (i.e. tensors with a single element). If `tensor` is multi-dimensional, then we can get a scalar from it by taking a weighted sum of all of the elements. The elements of `end_grad` are precisely the coefficients of our weighted sum. In other words, calling `tensor.backward(end_grad)` implicitly does the following:
#
# * Defines the value `L = (tensor * end_grad).sum()`
# * Backpropagates from `L` to all the other tensors in the graph before `tensor`
#
# So if `end_grad` is specified, it will be used as the `grad_out` argument in our first backward function. If `end_grad` is not specified, we assume `L = tensor.sum()`, i.e. `end_grad` is a tensor of all ones with the same shape as `tensor`.
#

# %% [markdown]
# ### Leaf nodes
#
# The `Tensor` object has an `is_leaf` property. Recall from the code earlier, we had:
#
# ```python
# class Tensor:
#
#     @property
#     def is_leaf(self):
#         '''Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html'''
#         if self.requires_grad and self.recipe and self.recipe.parents:
#             return False
#         return True
# ```
#
# In other words, leaf node tensors are any with either `requires_grad=False`, or which were *not* created by some operation on other tensors.
#
# In backprop, only the leaf nodes with `requires_grad=True` accumulate gradients. You can think of leaf nodes as being edges of the computational graph, i.e. nodes from which you can't move further backwards topologically.
#
# For example, suppose we have a neural network with a single linear layer called `layer`, and it produces `output` when we pass in `input`. Then:
#
# * `output` is not a leaf node, because it is the result of an operation on `layer.weight` and `input` (i.e. `recipe.parents` is not None)
# * `input` is a leaf node because it has `requires_grad=False` (this is the default for tensors) and it wasn't created from anything (i.e. `recipe` is None). So gradients will stop propagating when they get to `input`, but it won't store any gradients.
# * `layer.weight` is a leaf node because it wasn't created from anything (i.e. `recipe` is None). So gradients will stop propagating when they get to `layer.weight`, and it will store gradients (since `requires_grad=True`).
#
# ```python
# layer = torch.nn.Linear(3, 4)
# input = torch.ones(3)
# output = layer(input)
#
# print(layer.weight.is_leaf)       # -> True
# print(layer.weight.requires_grad) # -> True
#
# print(output.is_leaf)             # -> False
#
# print(input.is_leaf)              # -> True
# print(input.requires_grad)        # -> False
# ```
#
# In the computational graph in the next section, the only leaves are `a`, `b` and `c`.
#

# %% [markdown]
# ### Exercise - implement `backprop`
#
# ```yaml
# Difficulty: 🔴🔴🔴🔴🔴
# Importance: 🔵🔵🔵🔵🔵
#
# You should spend up to 30-40 minutes on this exercise.
#
# This exercise is the most conceptually important today. You should be willing to spend around 30 minutes on it. We've provided several dropdowns to help you.
# ```
#

# %% [markdown]
# Now, we get to the actual backprop function! Some code is provided below, which you should complete.
#
# If you want a challenge, you can try and implement it straight away, with out any help. However, because this is quite a challenging exercise, you can also use the dropdowns below. The first one gives you a sketch of the backpropagation algorithm, the second gives you a diagram which provides a bit more detail, and the third gives you the annotations for the function (so you just have to fill in the code). You are recommended to start by trying to implement it without help, but use the dropdowns (in order) if this is too difficult.
#
# We've also provided a few dropdowns to address specific technical errors that can arise from implementing this function. If you're having trouble, you can use these to help you debug.
#
# Either way, you're recommended to take some time with this function, as it's definitely the single most conceptually important exercise in the "Build Your Own Backpropagation Framework" section.
#

# %%
"""
## Backpropagation

Now all the pieces are in place to implement backpropagation. We need to:
- Loop over the nodes from right to left. At each node:
    - Call the backward function to transform the grad wrt output to the grad wrt input.
    - If the node is a leaf, write the grad to the grad field.
    - Otherwise, accumulate the grad into temporary storage.
"""


def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
    """Accumulates gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node:
        The rightmost node in the computation graph.
        If it contains more than one element, end_grad must be provided.
    end_grad:
        A tensor of the same shape as end_node.
        Set to 1 if not specified and end_node has only one element.
    """

    # find the grad of the last node
    if end_grad is not None:
        end_node_grad = end_grad.array
    else:
        end_node_grad = np.ones_like(end_node.array)

    topo = sorted_computational_graph(end_node)

    grads = {end_node: end_node_grad}

    # go in reverse topo order
    # starting from the end node
    for node in topo:
        node_grad = grads[node]

        # now that we're here, means that everything below
        # has been dealt with because we topo sorted this
        # means all the derivs flowing down from this node's children have been accumulated aka summed
        # means that we can "set" the grad property finally
        # but we only do it on the leafs and if they require grad
        # i had trouble understanding that leaves ≠ "input nodes"
        # https://chatgpt.com/share/e/6751bec9-3c28-800a-bca8-552313d2d6c4
        if node.requires_grad and node.is_leaf:
            node.grad = Tensor(node_grad)

        if not node.recipe or not node.recipe.parents:
            continue

        forward_func = node.recipe.func
        parents = node.recipe.parents

        # go through parents
        # and set their grads based on this node's grad
        args = node.recipe.args
        kwargs = node.recipe.kwargs
        for arg_pos, parent in parents.items():
            back_func = BACK_FUNCS.get_back_func(forward_func, arg_pos)
            partial = back_func(node_grad, node.array, *args, **kwargs)
            if parent not in grads:
                grads[parent] = partial
            else:
                grads[parent] += partial


# tests.test_backprop(Tensor)
# tests.test_backprop_branching(Tensor)
# tests.test_backprop_requires_grad_false(Tensor)
# tests.test_backprop_float_arg(Tensor)

# %% [markdown]
# <details>
# <summary>Dropdown #1 - sketch of algorithm</summary>
#
# You should iterate through the computational graph, in the order returned by your function (i.e. from right to left). For each tensor, you need to do two things:
#
# * If necessary, store the gradient in the `grad` field of the tensor. (This means you'll have to store the gradients in an external object, before setting them as attributes of the tensors.)
# * For each of the tensor's parents, store the gradients of those tensors for this particular path through the graph (this will require calling your backward functions, which you should get from the `BACK_FUNCS` object).
# </details>
#
# <details>
# <summary>Dropdown #2 - diagram of algorithm</summary>
#
# <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/backprop-2.png" width=800>
# </details>
#
# <details>
# <summary>Dropdown #3 - annotations</summary>
#
# Fill in the code beneath each annotation line that doesn't already have a line of code beneath it.
#
# Most annotations only require one line of code below them.
#
# ```python
# def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
#
#     # Get value of end_grad_arr
#
#     # Create dictionary 'grads' to store gradients
#
#     # Iterate through the computational graph, using your sorting function
#     for node in sorted_computational_graph(end_node):
#
#         # Get the outgradient from the grads dict
#
#         # If this node is a leaf & requires_grad is true, then store the gradient
#
#         # For all parents in the node:
#
#         # If node has a recipe, then we iterate through parents (which is a dict of {arg_posn: tensor})
#         for argnum, parent in node.recipe.parents.items():
#
#             # Get the backward function corresponding to the function that created this node
#
#             # Use this backward function to calculate the gradient
#
#             # Add the gradient to this node in the dictionary `grads`
# ```
# </details>
#

# %% [markdown]
# Specific technical issues:
#
# <details>
# <summary>Help - I get AttributeError: 'NoneType' object has no attribute 'func'</summary>
#
# This error is probably because you're trying to access `recipe.func` from the wrong node. Possibly, you're calling your backward functions using the parents nodes' `recipe.func`, rather than the node's `recipe.func`.
#
# To explain further, suppose your computational graph is simply:
#
# <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/Screenshot%202023-02-17%20174308.png" width=320>
#
# When you reach `b` in your backprop iteration, you should calculate the gradient wrt `a` (the only parent of `b`) and store it in your `grads` dictionary, as `grads[a]`. In order to do this, you need the backward function for `func1`, which is stored in the node `b` (recall that the recipe of a tensor can be thought of as a set of instructions for how that tensor was created).
# </details>
#
# <details>
# <summary>Help - I get AttributeError: 'numpy.ndarray' object has no attribute 'array'</summary>
#
# This might be because you've set `node.grad` to be an array, rather than a tensor. You should store gradients as tensors (think of PyTorch, where `tensor.grad` will have type `torch.Tensor`).
#
# It's fine to store numpy arrays in the `grads` dictionary, but when it comes time to set a tensor's grad attribute, you should use a tensor.
# </details>
#
# <details>
# <summary>Help - I get 'RuntimeError: bool value of Tensor with more than one value is ambiguous'.</summary>
#
# This error is probably because your computational graph function checks whether a tensor is in a list. The way these classes are compared for equality is a bit funky, and using sets rather than lists should make this error go away (i.e. checking whether a tensor is in a set should be fine).
# </details>
#

# %% [markdown]
# # 3️⃣ More forward & backward functions
#

# %% [markdown]
# > ## Learning Objectives
# >
# > * Implement more forward and backward functions, including for
# >   * Indexing
# >   * Non-differentiable functions
# >   * Matrix multiplication
#

# %% [markdown]
# Congrats on implementing backprop! The next thing we'll do is write implement a bunch of backward functions that we need to train our model at the end of the day, as well as ones that cover interesting cases.
#
# These should be just like your `log_back` and `multiply_back0`, `multiplyback1` examples earlier.
#

# %% [markdown]
# ***Note - some of these exercises can get a bit repetitive. About 60% of the value of these exercises was in the first 2 sections out of 5, and of the remaining 40%, not much of it is in this section! So you're welcome to skim through these exercises if you don't find them interesting.***
#

# %% [markdown]
# ## Non-Differentiable Functions
#
# For functions like `torch.argmax` or `torch.eq`, there's no sensible way to define gradients with respect to the input tensor. For these, we will still use `wrap_forward_fn` because we still need to unbox the arguments and box the result, but by passing `is_differentiable=False` we can avoid doing any unnecessary computation.
#
# We've given you this one as an example:
#


# %%
def _argmax(x: Arr, dim=None, keepdim=False):
    """Like torch.argmax."""
    return np.expand_dims(np.argmax(x, axis=dim), axis=([] if dim is None else dim))


argmax = wrap_forward_fn(_argmax, is_differentiable=False)

# a = Tensor([1.0, 0.0, 3.0, 4.0], requires_grad=True)
# b = a.argmax()
# print(a, b)
# assert not b.requires_grad
# assert b.recipe is None
# assert b.item() == 3

# %% [markdown]
# ## Single-Tensor Differentiable Functions
#

# %% [markdown]
# ### Exercise - `negative`
#
# ```yaml
# Difficulty: 🔴⚪⚪⚪⚪
# Importance: 🔵⚪⚪⚪⚪
#
# You should spend up to 5-10 minutes on this exercise.
# ```
#
# `torch.negative` just performs `-x` elementwise. Make your own version `negative` using `wrap_forward_fn`.
#


# %%
def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backward function for f(x) = -x elementwise."""
    return unbroadcast(-grad_out, x)


negative = wrap_forward_fn(np.negative)
BACK_FUNCS.add_back_func(np.negative, 0, negative_back)

# tests.test_negative_back(Tensor)

# %% [markdown]
# ### Exercise - `exp`
#
# ```yaml
# Difficulty: 🔴⚪⚪⚪⚪
# Importance: 🔵🔵⚪⚪⚪
#
# You should spend up to 5-10 minutes on this exercise.
# ```
#
# Make your own version of `torch.exp`. The backward function should express the result in terms of the `out` parameter - this more efficient than expressing it in terms of `x`.
#


# %%
def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return grad_out * out


exp = wrap_forward_fn(np.exp)
BACK_FUNCS.add_back_func(np.exp, 0, exp_back)

# tests.test_exp_back(Tensor)

# %% [markdown]
# ### Exercise - `reshape`
#
# ```yaml
# Difficulty: 🔴⚪⚪⚪⚪
# Importance: 🔵⚪⚪⚪⚪
#
# You should spend up to 5-10 minutes on this exercise.
# ```
#
# `reshape` is a bit more complicated than the many functions we've dealt with so far: there is an additional positional argument `new_shape`. Since it's not a `Tensor`, we don't need to think about differentiating with respect to it. Remember, `new_shape` is the argument that gets passed into the **forward function**, and we're trying to reverse this operation and return to the shape of the input.
#
# Depending how you wrote `wrap_forward_fn` and `backprop`, you might need to go back and adjust them to handle this. Or, you might just have to implement `reshape_back` and everything will work.
#
# Note that the output is a different shape than the input, but this doesn't introduce any additional complications.
#


# %%
def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    # this means that
    # first the input was reshaped to new shape, and then passed to function to produce out
    # so out.shape = new_shape
    # we know the gradients of
    # the graph looks like:
    # x -> reshape -> func -> out
    # we have already func_back'ed to get grad_out in this new shape
    # the gradients actual values will not change when reshaping_back
    # we just need to shape the gradients back to the shape of x
    return np.reshape(grad_out, newshape=x.shape)


reshape = wrap_forward_fn(np.reshape)
BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)

# tests.test_reshape_back(Tensor)

# %% [markdown]
# ### Exercise - `permute`
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵⚪⚪⚪⚪
#
# You should spend up to 10-15 minutes on this exercise.
# ```
#
# In NumPy, the equivalent of `torch.permute` is called `np.transpose`, so we will wrap that.
#


# %%
def invert_transposition(axes: tuple) -> tuple:
    """
    axes: tuple indicating a transition

    Returns: inverse of this transposition, i.e. the array `axes_inv` s.t. we have:
        np.transpose(np.transpose(x, axes), axes_inv) == x

    Some examples:
        (1, 0)    --> (1, 0)     # this is reversing a simple 2-element transposition
        (0, 2, 1) --> (0, 2, 1)  # also a 2-element transposition
        (1, 2, 0) --> (2, 0, 1)  # this is reversing the order of a 3-cycle
    """
    # note that this is not torch.transpose
    # this is permute
    # so the axes argument's length has to be equal to the number of dimensions in the tensor
    # when we do (2, 1, 0) means:
    #   - the 2nd dimension of the input is now the 0th dimension of the output
    #   - the 1st dimesion of the input is now the 1st dimension of the output
    #   - the 0th dimension of the input is now the 2nd dimension of the output
    # effectively, we need to find the index of 0, the index of 1 and index of 2 in the axes array
    # this will give us the index where the nth dimension "ended up"
    # we need to find the indices of 0, 1, ... n in the axes array
    # this gives us where those indices ended up
    # we then need to sort those indices we get using the original value

    # dealing with negative indices
    # if axes was (-1, -2)
    # means that the original axes[-1] went to axes[-2] and original axes[-2] went to axes[-1]
    # here axes will show up as (-1, -2)
    # again, note that every axis has to exist in this permutation tuple. so its length = len(input.shape)

    # so (-1, -2) means that the input was shape (a, b)
    # and became (b, a)
    # the 0th dimension ended up at 1, and the 1th dimension ended up at 0
    # we can convert all negative integers to 0+ integers
    # by doing: (-1 - idx) idx < 0 else idx

    # np.transpose's docstring gives a hint to how we can do this pretty easily: np.argsort(axes)

    axes_inv = []
    axes_non_negative = [(len(axes) + idx) if idx < 0 else idx for idx in axes]

    # print(axes, axes_non_negative)
    for i in range(len(axes_non_negative)):
        axes_inv.append(axes_non_negative.index(i))

    return axes_inv


def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    return np.transpose(grad_out, invert_transposition(axes))


BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
permute = wrap_forward_fn(np.transpose)

# tests.test_permute_back(Tensor)


# test negative
# x = Tensor(np.random.randint(0, 4, (3, 4)), requires_grad=True)
# y = x.T

# y.array, np.transpose(y.array, invert_transposition((-1, -2)))

# %% [markdown]
# <details>
# <summary>Help - I'm confused about how to implement this function.</summary>
#
# You should first define the function `invert_transposition`. A docstring is given below:
#
# ```python
# def invert_transposition(axes: tuple) -> tuple:
#     '''
#     axes: tuple indicating a transition
#
#     Returns: inverse of this transposition, i.e. the array `axes_inv` s.t. we have:
#         np.transpose(np.transpose(x, axes), axes_inv) == x
#
#     Some examples:
#         (1, 0)    --> (1, 0)     # this is reversing a simple 2-element transposition
#         (0, 2, 1) --> (0, 2, 1)  # also a 2-element transposition
#         (1, 2, 0) --> (2, 0, 1)  # this is reversing the order of a 3-cycle
#     '''
#     pass
# ```
#
# Once you've done this, you can define `permute_back` by transposing again, this time with the inversed transposition instructions.
# </details>

# %% [markdown]
# ### Exercise - `expand`
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵⚪⚪⚪⚪
#
# You should spend up to 15-20 minutes on this exercise.
# ```
#
# Implement your version of `torch.expand`.
#
# The backward function should just call `unbroadcast`.
#
# For the forward function, we will use `np.broadcast_to`. This function takes in an array and a target shape, and returns a version of the array broadcasted to the target shape using the rules of broadcasting we discussed in an earlier section. For example:
#

# # %%
# x = np.array([[1], [2], [3]])

# y = t.tensor([[1], [2], [3]])

# x.shape, y.shape, np.broadcast_to(x, (3, 3)), y.expand((3, -1))

# %%
# x = np.array([[1], [2], [3]])

# np.broadcast_to(x, (3, 3))  # x has shape (3, 1); broadcasting is done along rows


# %%
# t.tensor([1, 2]).expand((1, 1, 2))

# %% [markdown]
# The reason we can't just use `np.broadcast_to` and call it a day is that `torch.expand` supports -1 for a dimension size meaning "don't change the size". For example:
#

# %% [markdown]
# So when implementing `_expand`, you'll need to be a bit careful when constructing the shape to broadcast to.
#


# %%
def expand_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return unbroadcast(grad_out, x)


def _expand(x: Arr, new_shape) -> Arr:
    """
    Like torch.expand, calling np.broadcast_to internally.

    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    """
    # first we need to find the dimensions that have:
    # - been expanded (these will be 1s in the original shape)
    # - been kept the same

    maybe_changed_dims = list(new_shape[-len(x.shape) :])
    new_dims = list(new_shape[: -len(x.shape)])

    # now, if we have -1 in maybe_changed_dims, replace it with the corresponding dim from original tensor
    np_new_shape = new_dims + [
        dim if dim != -1 else x.shape[i] for i, dim in enumerate(maybe_changed_dims)
    ]
    new_array = np.broadcast_to(x, np_new_shape)
    return new_array


expand = wrap_forward_fn(_expand)
BACK_FUNCS.add_back_func(_expand, 0, expand_back)

# tests.test_expand(Tensor)
# tests.test_expand_negative_length(Tensor)

# %% [markdown]
# <details>
# <summary>Help - I'm not sure how to construct the shape.</summary>
#
# If `new_shape` contains no -1s, then you're done. If it does contain -1s, you want to replace those with the appropriate values from `x.shape`.
#
# For example, if `a.shape = (5,)`, and `new_shape = (3, 2, -1)`, you want the actual shape passed into `np.broadcast_to` to be `(3, 2, 5)`.
# </details>
#

# %% [markdown]
# ### Exercise - `sum`
#
# ```c
# Difficulty: 🔴🔴🔴🔴⚪
# Importance: 🔵🔵⚪⚪⚪
#
# You should spend up to 20-30 minutes on this exercise.
# This one can be a bit tricky, so don't be afraid to look at the hint.
# ```
#
# The output can also be smaller than the input, such as when calling `torch.sum`. Implement your own `torch.sum` and `sum_back`.
#
# Note, if you get weird exceptions that you can't explain, and these exceptions don't even go away when you use the solutions provided, this probably means that your implementation of `wrap_forward_fn` was wrong in a way which wasn't picked up by the tests. You should return to this function and try to fix it (or just use the solution).
#
# <details>
# <summary>Help - I get the error "Encountered error when running `backward` in the test for nonscalar grad_out."</summary>
#
# This error is likely due to the fact that you're expanding your tensor in a way that doesn't refer to the dimensions being summed over (i.e. the `dim` argument).
#
# Remember that in the previous exercise we assumed that the tensors were broadcastable with each other, and our functions could just internally call `np.broadcast_to` as a result. But here, one tensor is the sum over another tensor's dimensions, and if `keepdim=False` then they might not broadcast. For instance, if `x.shape = (2, 5)`, `out = x.sum(dim=1)` has shape `(2,)` and `grad_out.shape = (2,)`, then the tensors `grad_out` and `x` are not broadcastable (in fact, this is exactly the test case that's causing the error here).
#
# How can you carefully handle the case where `keepdim=False` and `dim` doesn't just refer to dimensions at the start of the tensor? (Hint - try and use `np.expand_dims`).
#
# </details>


# %%
def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False):
    """Basic idea: repeat grad_out over the dims along which x was summed"""
    # we took an input x
    # we summed it along some dimension

    # how to find the dims we summed along:
    # If we kept dims, we just need to repeat and match with the original tensor using expand
    # If we didn't keep dims, we need to find the "missing" dims from the new shape, and create a new shape, then expand to that shape

    if dim == None or keepdim == True:
        return _expand(grad_out, x.shape)
    else:
        # means keepdims was False and dim was provided
        grad_out = np.expand_dims(grad_out, axis=dim)

        return _expand(grad_out, x.shape)


def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    """Like torch.sum, calling np.sum internally."""
    return np.sum(x, axis=dim, keepdims=keepdim)


sum = wrap_forward_fn(_sum)
BACK_FUNCS.add_back_func(_sum, 0, sum_back)

# tests.test_sum_keepdim_false(Tensor)
# tests.test_sum_keepdim_true(Tensor)
# tests.test_sum_dim_none(Tensor)
# tests.test_sum_nonscalar_grad_out(Tensor)

# %% [markdown]
# ### Exercise - Indexing
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵⚪⚪⚪
#
# You should spend up to 15-20 minutes on this exercise.
# ```
#
# In its full generality, indexing a `torch.Tensor` is really complicated and there are quite a few cases to handle separately.
#
# We only need two cases today:
# - The index is an integer or tuple of integers.
# - The index is a tuple of (array or Tensor) representing coordinates. Each array is 1D and of equal length. Some coordinates may be repeated. This is [Integer array indexing](https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing).
#     - For example, to select the five elements at (0, 0), (1,0), (0, 1), (1, 2), and (0, 0), the index would be the tuple `(np.array([0, 1, 0, 1, 0]), np.array([0, 0, 1, 2, 0]))`.
#
# Note, in `_getitem` you'll need to deal with one special case: when `index` is of type signature `tuple[Tensor]`. If not for this case, `return x[index]` would suffice for this function. You should define a `coerce_index` function to deal with this particular case; we've provided a docstring for this purpose.
#

# # %%
# np.array(np.array(4))

# %%
Index = int | tuple[int, ...] | tuple[Arr] | tuple[Tensor]


def coerce_index(index: Index) -> int | tuple[int, ...] | tuple[Arr]:
    """
    If index is of type signature `tuple[Tensor]`, converts it to `tuple[Arr]`.
    """
    if type(index) == int:
        return index

    return tuple([x.array if isinstance(x, Tensor) else x for x in index])


def _getitem(x: Arr, index: Index) -> Arr:
    """Like x[index] when x is a torch.Tensor."""
    return x[coerce_index(index)]


def getitem_back(grad_out: Arr, out: Arr, x: Arr, index: Index):
    """
    Backwards function for _getitem.

    Hint: use np.add.at(a, indices, b)
    This function works just like a[indices] += b, except that it allows for repeated indices.
    """
    # we took an input x
    # ran getitem on it to get output out
    # now we didn't actually do anything to the input
    # so the gradients should just flow back without any change
    # but because indices can be repeated, means that 1 value can be causing multiple values to change in the output
    # so x change in 1 value could be causing nx changes in the output
    # hence we need to add grads at those indices

    # initialize zeros
    grad_in = np.zeros_like(x)

    np.add.at(grad_in, coerce_index(index), grad_out)

    return grad_in


getitem = wrap_forward_fn(_getitem)
BACK_FUNCS.add_back_func(_getitem, 0, getitem_back)

# tests.test_coerce_index(coerce_index, Tensor)
# tests.test_getitem_int(Tensor)
# tests.test_getitem_tuple(Tensor)
# tests.test_getitem_integer_array(Tensor)
# tests.test_getitem_integer_tensor(Tensor)

# %% [markdown]
# <details>
# <summary>Help - I'm confused about how to implement getitem_back.</summary>
#
# If no coordinates were repeated, we could just assign the grad for each input element to be the grad at the corresponding output position, or 0 if that input element didn't appear.
#
# Because of the potential for repeat coordinates, we need to sum the grad from each corresponding output position.
#
# Initialize an array of zeros of the same shape as x, and then write in the appropriate elements using `np.add.at`.
# </details>
#

# %% [markdown]
# ### elementwise add, subtract, divide
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵⚪⚪⚪
#
# You should spend up to 10-15 minutes on this exercise.
# ```
#
# These are exactly analogous to the multiply case. Note that Python and NumPy have the notion of "floor division", which is a truncating integer division as in `7 // 3 = 2`. You can ignore floor division: - we only need the usual floating point division which is called "true division".
#
# Use lambda functions to define and register the backward functions each in one line. If you're confused, you can click on the expander below to reveal the first one.
#

# %% [markdown]
# <details>
# <summary>Reveal the first one:</summary>
#
# ```python
# BACK_FUNCS.add_back_func(np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))
# ```
# </details>
#

# %%
add = wrap_forward_fn(np.add)
subtract = wrap_forward_fn(np.subtract)
true_divide = wrap_forward_fn(np.true_divide)

# Your code here - add to the BACK_FUNCS object

BACK_FUNCS.add_back_func(
    np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x)
)
BACK_FUNCS.add_back_func(
    np.add, 1, lambda grad_out, out, x, y: unbroadcast(grad_out, y)
)

BACK_FUNCS.add_back_func(
    np.subtract, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x)
)
BACK_FUNCS.add_back_func(
    np.subtract, 1, lambda grad_out, out, x, y: unbroadcast(-grad_out, y)
)

BACK_FUNCS.add_back_func(
    np.true_divide, 0, lambda grad_out, out, x, y: unbroadcast(grad_out / y, x)
)
BACK_FUNCS.add_back_func(
    np.true_divide,
    1,
    lambda grad_out, out, x, y: unbroadcast((-grad_out * x) / (y**2), y),
)


# %%
# tests.test_add_broadcasted(Tensor)
# tests.test_subtract_broadcasted(Tensor)
# tests.test_truedivide_broadcasted(Tensor)

# %% [markdown]
# ## In-Place Operations
#
# Supporting in-place operations introduces substantial complexity and generally doesn't help performance that much. The problem is that if any of the inputs used in the backward function have been modified in-place since the forward pass, then the backward function will incorrectly calculate using the modified version.
#
# PyTorch will warn you when this causes a problem with the error "RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.".
#
# You can implement the warning in the bonus section but for now your system will silently compute the wrong gradients - user beware!
#
# (note - you don't have to fill anything in here; just run the cell)
#


# %%
def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
    """Like torch.add_. Compute x += other * alpha in-place and return tensor."""
    np.add(x.array, other.array * alpha, out=x.array)
    return x


# def safe_example():
#     """This example should work properly."""
#     a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
#     b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
#     a.add_(b)
#     c = a * b
#     c.sum().backward()
#     assert a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0])
#     assert b.grad is not None and np.allclose(b.grad.array, [2.0, 4.0, 6.0, 8.0])


# def unsafe_example():
#     """This example is expected to compute the wrong gradients."""
#     a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
#     b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
#     c = a * b
#     a.add_(b)
#     c.sum().backward()
#     if a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0]):
#         print("Grad wrt a is OK!")
#     else:
#         print("Grad wrt a is WRONG!")
#     if b.grad is not None and np.allclose(b.grad.array, [0.0, 1.0, 2.0, 3.0]):
#         print("Grad wrt b is OK!")
#     else:
#         print("Grad wrt b is WRONG!")


# safe_example()
# unsafe_example()

# %% [markdown]
# ## Mixed Scalar-Tensor Operations
#
# You may have been wondering why our `Tensor` class has to define both `__mul__` and `__rmul__` magic methods.
#
# Without `__rmul__` defined, executing `2 * a` when `a` is a `Tensor` would try to call `2.__mul__(a)`, and the built-in class `int` would be confused about how to handle this.
#
# Since we have defined `__rmul__` for you at the start, and you implemented multiply to work with floats as arguments, the following should "just work".
#

# %%
# a = Tensor([0, 1, 2, 3], requires_grad=True)
# (a * 2).sum().backward()
# b = Tensor([0, 1, 2, 3], requires_grad=True)
# (2 * b).sum().backward()
# assert a.grad is not None
# assert b.grad is not None
# assert np.allclose(a.grad.array, b.grad.array)

# %% [markdown]
# ### Exercise - `max`
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵⚪⚪⚪
#
# You should spend up to 10-15 minutes on this exercise.
# ```
#
# Since this is an elementwise function, we can think about the scalar case. For scalar $x$, $y$, the derivative for $\max(x, y)$ wrt $x$ is 1 when $x > y$ and 0 when $x < y$. What should happen when $x = y$?
#
# Intuitively, since $\max(x, x)$ is equivalent to the identity function which has a derivative of 1 wrt $x$, it makes sense for the sum of our partial derivatives wrt $x$ and $y$ to also therefore total 1. The convention used by PyTorch is to split the derivative evenly between the two arguments. We will follow this behavior for compatibility, but it's just as legitimate to say it's 1 wrt $x$ and 0 wrt $y$, or some other arbitrary combination that sums to one.
#

# %% [markdown]
# <details>
# <summary>Help - I'm not sure how to implement this function.</summary>
#
# Try returning `grad_out * bool_sum`, where `bool_sum` is an array constructed from the sum of two boolean arrays.
#
# You can alternatively use `np.where`.
# </details>
#
# <details>
# <summary>Help - I'm passing the first test but not the second.</summary>
#
# This probably means that you haven't implemented `unbroadcast`. You'll need to do this, to get `grad_out` into the right shape before you use it in `np.where`.
# </details>
#

# %%

# x = np.random.randint(0, 3, (3, 4))
# y = np.array([1.5])
# x, y, np.maximum(x, y)


# %%
def maximum_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    """Backwards function for max(x, y) wrt x."""
    # x and y could have been broadcasted to the shape of out
    # first broadcast both to the shape of out
    # then return grad bsaed on true, false of x > y
    x_broadcasted = np.broadcast_to(x, out.shape)
    y_broadcasted = np.broadcast_to(y, out.shape)

    g = np.ones_like(grad_out)

    g[x_broadcasted > y_broadcasted] = 1
    g[x_broadcasted < y_broadcasted] = 0
    g[x_broadcasted == y_broadcasted] = 0.5

    g = grad_out * g

    return unbroadcast(g, x)


def maximum_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    """Backwards function for max(x, y) wrt y."""
    # x and y could have been broadcasted to the shape of out
    # first broadcast both to the shape of out
    # then return grad bsaed on true, false of x > y
    x_broadcasted = np.broadcast_to(x, grad_out.shape)
    y_broadcasted = np.broadcast_to(y, grad_out.shape)

    g = np.ones_like(grad_out)

    g[y_broadcasted > x_broadcasted] = 1
    g[y_broadcasted < x_broadcasted] = 0
    g[y_broadcasted == x_broadcasted] = 0.5

    g = grad_out * g

    return unbroadcast(g, y)


maximum = wrap_forward_fn(np.maximum)

BACK_FUNCS.add_back_func(np.maximum, 0, maximum_back0)
BACK_FUNCS.add_back_func(np.maximum, 1, maximum_back1)

# tests.test_maximum(Tensor)
# tests.test_maximum_broadcasted(Tensor)

# %% [markdown]
# ### Exercise - functional `ReLU`
#
# ```yaml
# Difficulty: 🔴🔴⚪⚪⚪
# Importance: 🔵🔵⚪⚪⚪
#
# You should spend up to 10-15 minutes on this exercise.
# ```
#
# A simple and correct ReLU function can be defined in terms of your maximum function. Note the PyTorch version also supports in-place operation, which we are punting on for now.
#
# Again, at $x = 0$ your derivative could reasonably be anything between 0 and 1 inclusive, but we've followed PyTorch in making it 0.5.
#


# %%
def relu(x: Tensor) -> Tensor:
    """Like torch.nn.function.relu(x, inplace=False)."""

    return maximum(x, 0)


# tests.test_relu(Tensor)

# %% [markdown]
# ### Exercise - 2D `matmul`
#
# ```yaml
# Difficulty: 🔴🔴🔴🔴⚪
# Importance: 🔵🔵🔵⚪⚪
#
# You should spend up to 20-25 minutes on this exercise.
# ```
#
# Implement your version of `torch.matmul`, restricting it to the simpler case where both inputs are 2D.
#

# %%
# x = np.random.randint(0, 4, (4, 2))
# y = np.random.randint(1, 4, (2, 3))
# y_expanded = np.expand_dims(y, axis=1)
# x_expanded = np.expand_dims(x.transpose(), axis=1)
# mul = x @ y
# ty = mul * y_expanded
# tx = mul.transpose() * x_expanded
# # x, y, y_expanded, mul, t, np.reshape(np.sum(t, axis=-1).transpose(), x.shape)
# y, x_expanded, mul, tx, mul.shape, np.reshape(
#     np.sum(tx, axis=-1), y.shape
# ), x_expanded.shape, tx.shape

# %%


# %%


def _matmul2d(x: Arr, y: Arr) -> Arr:
    """Matrix multiply restricted to the case where both inputs are exactly 2D."""
    return x @ y


def matmul2d_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    # grad of x depends on rows of y
    # nth column of x = grad_out * nth row of y, then summed along rows
    # apparently a very slick solution exists: grad_out * y.T lol.
    # y_expanded = np.expand_dims(y, axis=1)
    # t = grad_out * y_expanded
    # return np.reshape(np.sum(t, axis=-1).transpose(), x.shape)

    return grad_out @ y.T


def matmul2d_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    # x_expanded = np.expand_dims(x.transpose(), axis=1)
    # t = grad_out.transpose() * x_expanded
    # return np.reshape(np.sum(t, axis=-1), y.shape)
    return x.T @ grad_out


matmul = wrap_forward_fn(_matmul2d)
BACK_FUNCS.add_back_func(_matmul2d, 0, matmul2d_back0)
BACK_FUNCS.add_back_func(_matmul2d, 1, matmul2d_back1)

# tests.test_matmul2d(Tensor)

# %% [markdown]
# <details>
# <summary>Help - I'm confused about <code>matmul2d_back</code>!</summary>
#
# Let $X$, $Y$ and $M$ denote the variables `x`, `y` and `out`, so we have the matrix relation $M = XY$. The object `grad_out` is a tensor with elements `grad_out[p, q]` $ = \frac{\partial L}{\partial M_{p q}}$.
#
# The output of `matmul2d_back0` should be the gradient of $L$ wrt $X$, i.e. it should have elements $\frac{\partial L}{\partial X_{i j}}$. Can you write this in terms of the elements of `x`, `y`, `out` and `grad_out`?
# </details>
#
# <details>
# <summary>Help - I'm still stuck on <code>matmul2d_back</code>.</summary>
#
# We can write $\frac{\partial L}{\partial X_{i j}}$ as:
#
# $$
# \begin{aligned}
# \frac{\partial L}{\partial X_{i j}} &=\sum_{pq} \frac{\partial L}{\partial M_{p q}} \frac{\partial M_{p q}}{\partial X_{i j}} \\
# &=\sum_{pq} \left[\text{ grad\_out }\right]_{p q} \frac{\partial (\sum_r X_{p r} Y_{r q})}{\partial X_{i j}} \quad\quad \text{ because } M_{pq} = \sum_r X_{pr} Y_{rq} \\
# &=\sum_{pqr} \left[\text{ grad\_out }\right]_{p q}  \frac{\partial X_{p r}}{\partial X_{i j}} Y_{rq} \\
# &=\sum_{q} \left[\text{ grad\_out }\right]_{iq} Y_{j q} \quad\quad \text{because } \frac{\partial{X_{pr}}}{X_{ij}} = 1 \text{ if } (p, r) = (i, j), \text{ else } 0 \\
# &=\sum_{q} \left[\text{ grad\_out }\right]_{iq} Y^T_{qj} \\
# &= \left[\text{ grad\_out } \times Y^{\top}\right]_{ij}
# \end{aligned}
# $$
#
#
# In other words, the `x.grad` attribute should be is `grad_out @ y.T`.
#
# You can calculate the gradient wrt `y` in a similar way - we leave this as an exercise for the reader.
# </details>

# %% [markdown]
# # 4️⃣ Putting everything together
#

# %% [markdown]
# > ## Learning Objectives
# >
# > * Complete the process of building up a neural network from scratch and training it via gradient descent.
#

# %% [markdown]
# ## Exercise - build your own `nn.Parameter`
#
# ```yaml
# Difficulty: 🔴🔴⚪⚪⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to 10-15 minutes on this exercise.
# ```
#
# We've now written enough backwards passes that we can go up a layer and write our own `nn.Parameter` and `nn.Module`.
# We don't need much for `Parameter`. It is itself a `Tensor`, shares storage with the provided `Tensor` and requires_grad is `True` by default - that's it!
#

# %%


# %%
class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True, name=""):
        """Share the array with the provided tensor."""
        self.name = name
        return super().__init__(tensor.array, requires_grad=requires_grad)

    def __repr__(self):
        return f"Parameter containing:\n" + super().__repr__()


# x = Tensor([1.0, 2.0, 3.0])
# p = Parameter(x)
# assert p.requires_grad
# assert p.array is x.array
# assert (
#     repr(p)
#     == "Parameter containing:\nTensor(array([1., 2., 3.], dtype=float32), requires_grad=True)"
# )
# x.add_(Tensor(np.array(2.0)))
# assert np.allclose(
#     p.array, np.array([3.0, 4.0, 5.0])
# ), "in-place modifications to the original tensor should affect the parameter"

# %% [markdown]
# ## Exercise - build your own `nn.Module`
#
# ```yaml
# Difficulty: 🔴🔴🔴🔴⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to 25-30 minutes on this exercise.
# ```
#
# `nn.Module` is like `torch.Tensor` in that it has a lot of functionality, most of which we don't care about today. We will just implement enough to get our network training.
#
# Implement the indicated methods (i.e. the ones which are currently just `pass`). We've defined `_modules` and `_parameters` dict for you (note that, because this isn't a dataclass, we need to manually define them inside the `__init__` method).
#
# Tip: you can bypass `__getattr__` by accessing `self.__dict__` inside a method.
#
# *Note - some of these methods are difficult and non-obvious to implement, so don't worry if you need to look at the solutions.*
#


# %%
class Module:
    _modules: Dict[str, "Module"]
    _parameters: Dict[str, Parameter]

    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def modules(self):
        """Return the direct child modules of this module."""
        return self.__dict__["_modules"].values()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Return an iterator over Module parameters.

        recurse: if True, the iterator includes parameters of submodules, recursively.
        """
        for p in self._parameters.values():
            yield p

        if recurse:
            for m in self._modules.values():
                yield from m.parameters()

    def __setattr__(self, key: str, val: Any) -> None:
        """
        If val is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call __setattr__ from the superclass.
        """
        if isinstance(val, Parameter):
            self._parameters[key] = val
        if isinstance(val, Module):
            self._modules[key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Union[Parameter, "Module"]:
        """
        If key is in _parameters or _modules, return the corresponding value.
        Otherwise, raise KeyError.
        """
        if key in self._parameters:
            return self._parameters[key]
        elif key in self._modules:
            return self._modules[key]

        raise KeyError(key)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError("Subclasses must implement forward!")

    def __repr__(self):
        def _indent(s_, numSpaces):
            return re.sub("\n", "\n" + (" " * numSpaces), s_)

        lines = [
            f"({key}): {_indent(repr(module), 2)}"
            for key, module in self._modules.items()
        ]
        return "".join(
            [
                self.__class__.__name__ + "(",
                "\n  " + "\n  ".join(lines) + "\n" if lines else "",
                ")",
            ]
        )


class TestInnerModule(Module):
    def __init__(self):
        super().__init__()
        self.param1 = Parameter(Tensor([1.0]))
        self.param2 = Parameter(Tensor([2.0]))


class TestModule(Module):
    def __init__(self):
        super().__init__()
        self.inner = TestInnerModule()
        self.param3 = Parameter(Tensor([3.0]))


# mod = TestModule()
# assert list(mod.modules()) == [mod.inner]
# assert list(mod.parameters()) == [
#     mod.param3,
#     mod.inner.param1,
#     mod.inner.param2,
# ], "parameters should come before submodule parameters"
# print("Manually verify that the repr looks reasonable:")
# print(mod)

# %% [markdown]
# ## Build Your Own Linear Layer
#
# ```yaml
# Difficulty: 🔴🔴⚪⚪⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to 20-25 minutes on this exercise.
# ```
#
# You may have a `Linear` written already that you can adapt to use our own `Parameter`, `Module`, and `Tensor`. If your `Linear` used `einsum`, use a `matmul` instead. You can implement a backward function for `einsum` in the bonus section.
#


# %%
class Linear(Module):
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, in_features: int, out_features: int, bias=True, name=""):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        # x will be n x in_features
        # so weight has to be in_features x out_features
        sf = in_features**-0.5
        self.name = name

        weight = sf * Tensor(2 * np.random.rand(out_features, in_features) - 1)

        self.weight = Parameter(weight, name=f"{name}: {in_features}->{out_features}")

        self.in_features = in_features
        self.out_features = out_features

        if bias:
            bias = sf * Tensor(
                2
                * np.random.rand(
                    out_features,
                )
                - 1
            )
            self.bias = Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        m = x @ self.weight.T

        # print(m.shape)
        # print(self.extra_repr())

        if self.bias is not None:
            m = m + self.bias

        return m

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# linear = Linear(3, 4)
# assert isinstance(linear.weight, Tensor)
# assert linear.weight.requires_grad

# input = Tensor([[1.0, 2.0, 3.0]])
# output = linear(input)
# assert output.requires_grad

# expected_output = input @ linear.weight.T + linear.bias
# np.testing.assert_allclose(output.array, expected_output.array)

# print("All tests for `Linear` passed!")

# %% [markdown]
# Finally, for the sake of completeness, we'll define a `ReLU` module:
#


# %%
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


# %% [markdown]
# Now we can define a MLP suitable for classifying MNIST, with zero PyTorch dependency!
#


# %%
class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(28 * 28, 64, name="linear1")
        self.linear2 = Linear(64, 64, name="linear2")
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.output = Linear(64, 10, name="output")

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape((x.shape[0], 28 * 28))
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.output(x)
        return x


# %% [markdown]
# ## Build Your Own Cross-Entropy Loss
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵🔵⚪⚪
#
# You should spend up to 10-15 minutes on this exercise.
# ```
#
# Make use of your integer array indexing to implement `cross_entropy`. See the documentation page [here](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
#
# Note - if you have a tensor `X` of shape `(a, b)`, and you want to take the `Y[i]`-th element of each row (where `Y` is a tensor of length `a`), the easiest way to do this is with:
#
# ```python
# X[range(a), Y]
# ```
#
# since this is read as:
#
# ```python
# X[0, Y[0]], X[1, Y[1]], ..., X[a-1, Y[a-1]]
# ```
#
# In place of `range` here, you should use the function `arange` which we've provided for you (this works in exactly the same way, but it returns a `Tensor` instead of a `range` object).
#


# %%
def cross_entropy(logits: Tensor, true_labels: Tensor) -> Tensor:
    """Like torch.nn.functional.cross_entropy with reduction='none'.

    logits: shape (batch, classes)
    true_labels: shape (batch,). Each element is the index of the correct label in the logits.

    Return: shape (batch, ) containing the per-example loss.
    """
    n_batches, n_classes = logits.shape
    predicted = logits[arange(0, n_batches), true_labels]
    return -(predicted.exp() / logits.exp().sum(1)).log()


# tests.test_cross_entropy(Tensor, cross_entropy)

# %%


# %% [markdown]
# ## Build your own `NoGrad` context manager
#
# ```yaml
# Difficulty: 🔴🔴⚪⚪⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to 10-15 minutes on this exercise.
# ```
#
# The last thing our backpropagation system needs is the ability to turn it off completely like `torch.no_grad`.
#
# Below, you should implement the `NoGrad` context manager so that it reads and writes the `grad_tracking_enabled` flag from the top of the file.
#
# You should use the global variable `grad_tracking_enabled` **global variables**  for this. To make sure you're accessing the global variable rather than just a local variable, you can use the `global` keyword. In general, using mutable global variables is not ideal because multiple threads will be a problem, but we will leave that for another day.
#


# %%
class NoGrad:
    """Context manager that disables grad inside the block. Like torch.no_grad."""

    was_enabled: bool

    def __enter__(self):
        """
        Method which is called whenever the context manager is entered, i.e. at the
        start of the `with NoGrad():` block.
        """
        global grad_tracking_enabled
        self.was_enabled = grad_tracking_enabled
        grad_tracking_enabled = False

    def __exit__(self, type, value, traceback):
        """
        Method which is called whenever we exit the context manager.
        """
        global grad_tracking_enabled
        grad_tracking_enabled = self.was_enabled


# %% [markdown]
# <details>
# <summary>Help - I'm not sure what to do here.</summary>
#
# You should put `global grad_tracking_enabled` at the top of both methods.
#
# In the `__enter__` method, you should disable gradient tracking (i.e. set the global variable to False).
#
# In the `__exit__` method, you should set gradient tracking to whatever its global value was *before* you entered the context manager (this is why we need to use the `was_enabled` variable, so you can record what the global value was before you entered the context manager).
# </details>

# %% [markdown]
# ## Training Your Network
#
# We've already looked at data loading and training loops earlier in the course, so we'll provide a minimal version of these today as well as the data loading code.
#

# %%
# train_loader, test_loader = get_mnist()
# visualize(train_loader)

# %% [markdown]
# And here's a basic optimizer & training/testing loop:
#


# %%
# class SGD:
#     def __init__(self, params: Iterable[Parameter], lr: float):
#         """Vanilla SGD with no additional features."""
#         self.params = list(params)
#         self.lr = lr
#         self.b = [None for _ in self.params]

#     def zero_grad(self) -> None:
#         for p in self.params:
#             p.grad = None

#     def step(self) -> None:
#         with NoGrad():
#             for i, p in enumerate(self.params):
#                 assert isinstance(p.grad, Tensor)
#                 p.add_(p.grad, -self.lr)


# def train(
#     model: MLP,
#     train_loader: DataLoader,
#     optimizer: SGD,
#     epoch: int,
#     train_loss_list: Optional[list] = None,
# ):
#     print(f"Epoch: {epoch}")
#     progress_bar = tqdm(enumerate(train_loader))
#     for batch_idx, (data, target) in progress_bar:
#         data = Tensor(data.numpy())
#         target = Tensor(target.numpy())
#         optimizer.zero_grad()
#         output = model(data)
#         loss = cross_entropy(output, target).sum() / len(output)
#         loss.backward()
#         progress_bar.set_description(f"Train set: Avg loss: {loss.item():.3f}")
#         optimizer.step()
#         if train_loss_list is not None:
#             train_loss_list.append(loss.item())


# def test(model: MLP, test_loader: DataLoader, test_loss_list: Optional[list] = None):
#     test_loss = 0
#     correct = 0
#     with NoGrad():
#         for data, target in test_loader:
#             data = Tensor(data.numpy())
#             target = Tensor(target.numpy())
#             output: Tensor = model(data)
#             test_loss += cross_entropy(output, target).sum().item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += (pred == target.reshape(pred.shape)).sum().item()
#     n_data = len(test_loader.dataset)  # type: ignore
#     test_loss /= n_data
#     print(
#         f"Test set:  Avg loss: {test_loss:.4f}, Accuracy: {correct}/{n_data} ({correct / n_data:.1%})"
#     )
#     if test_loss_list is not None:
#         test_loss_list.append(test_loss)


# %% [markdown]
# ### Training Loop
#
# To finish the day, let's see if everything works correctly and our MLP learns to classify MNIST. It's normal to encounter some bugs and glitches at this point - just go back and fix them until everything runs.
#

# %%
# num_epochs = 5
# model = MLP()
# start = time.time()
# train_loss_list = []
# test_loss_list = []
# optimizer = SGD(model.parameters(), 0.01)
# for epoch in range(num_epochs):
#     train(model, train_loader, optimizer, epoch, train_loss_list)
#     test(model, test_loader, test_loss_list)
#     optimizer.step()
# print(f"\nCompleted in {time.time() - start: .2f}s")

# # %%
# line(
#     train_loss_list,
#     yaxis_range=[0, max(train_loss_list) + 0.1],
#     labels={"x": "Batches seen", "y": "Cross entropy loss"},
#     title="ConvNet training on MNIST",
#     width=800,
#     hovermode="x unified",
#     template="ggplot2",  # alternative aesthetic for your plots (-:
# )

# %% [markdown]
# Note - this training loop (if done correctly) will look to the one we used in earlier sections is that we're using SGD rather than Adam. You can try adapting your Adam code from the previous day's exercises, and get the same results as you have in earlier sections.
#

# %% [markdown]
# # 5️⃣ Bonus
#

# %% [markdown]
# Congratulations on finishing the day's main content! Here are a few more bonus things for you to explore.
#

# %% [markdown]
# ### In-Place Operation Warnings
#
# The most severe issue with our current system is that it can silently compute the wrong gradients when in-place operations are used. Have a look at how [PyTorch handles it](https://pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd) and implement a similar system yourself so that it either computes the right gradients, or raises a warning.
#

# %% [markdown]
# ### In-Place `ReLU`
#
# Instead of implementing ReLU in terms of maximum, implement your own forward and backward functions that support `inplace=True`.
#

# %% [markdown]
# ### Backward for `einsum`
#
# Write the backward pass for your equivalent of `torch.einsum`.
#

# %% [markdown]
# ### Reuse of Module during forward
#
# Consider the following MLP, where the same `nn.ReLU` instance is used twice in the forward pass. Without running the code, explain whether this works correctly or not with reference to the specifics of your implementation.
#
# ```python
# class MyModule(Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = Linear(28*28, 64)
#         self.linear2 = Linear(64, 64)
#         self.linear3 = Linear(64, 10)
#         self.relu = ReLU()
#     def forward(self, x):
#         x = self.relu(self.linear1(x))
#         x = self.relu(self.linear2(x))
#         return self.linear3(x)
# ```
#
# <details>
# <summary>Answer (what you should find)</summary>
#
# This implementation will work correctly.
#
# The danger of reusing modules is that you'd be creating a cyclical computational graph (because the same parameters would appear twice), but the `ReLU` module doesn't have any parameters (or any internal state), so this isn't a problem. It's effectively just a wrapper for the `relu` function, and you could replace `self.relu` with applying the `relu` function directly without changing the model's behaviour.
#
# This is slightly different if we're thinking about adding **hooks** to our model. Hooks are functions that are called during the forward or backward pass, and they can be used to inspect the state of the model during training. We generally want each hook to be associated with a single position in the model, rather than being called at two different points.
# </details>
#

# %% [markdown]
# ### Convolutional layers
#
# Now that you've implemented a linear layer, it should be relatively straightforward to take your convolutions code from day 2 and use it to make a convolutional layer. How much better performance do you get on the MNIST task once you replace your first two linear layers with convolutions?
#

# %% [markdown]
# ### ResNet Support
#
# Make a list of the features that would need to be implemented to support ResNet inference, and training. It will probably take too long to do all of them, but pick some interesting features to start implementing.
#

# %% [markdown]
# ### Central Difference Checking
#
# Write a function that compares the gradients from your backprop to a central difference method. See [Wikipedia](https://en.wikipedia.org/wiki/Finite_difference) for more details.
#

# %% [markdown]
# ### Non-Differentiable Function Support
#
# Your `Tensor` does not currently support equivalents of `torch.all`, `torch.any`, `torch.floor`, `torch.less`, etc. which are non-differentiable functions of Tensors. Implement them so that they are usable in computational graphs, but gradients shouldn't flow through them (their contribution is zero).
#

# %% [markdown]
# ### Differentiation wrt Keyword Arguments
#
# In the real PyTorch, you can sometimes pass tensors as keyword arguments and differentiation will work, as in `t.add(other=t.tensor([3,4]), input=t.tensor([1,2]))`. In other similar looking cases like `t.dot`, it raises an error that the argument must be passed positionally. Decide on a desired behavior in your system and implement and test it.
#

# %% [markdown]
# ### `torch.stack`
#
# So far we've registered a separate backwards for each input argument that could be a Tensor. This is problematic if the function can take any number of tensors like `torch.stack` or `numpy.stack`. Think of and implement the backward function for stack. It may require modification to your other code.
#
