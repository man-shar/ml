# %% [markdown]
# # [0.2] - CNNs & ResNets
#

# %% [markdown]
# Colab: [exercises](https://colab.research.google.com/drive/1sZpsnjN7gI4EucRcD0mCafN5Vg1DPOnm) | [solutions](https://colab.research.google.com/drive/1LDv8fB47OPKPp4C83CO_gTzaUE2bR-Ve)
#
# ARENA 3.0 [Streamlit page](https://arena3-chapter0-fundamentals.streamlit.app/[0.2]_CNNs_&_ResNets)
#
# Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-uk/shared_invite/zt-2noug8mpy-TRYbCnc3pzj7ITNrZIjKww), and ask any questions on the dedicated channels for this chapter of material.
#
# To view solutions for the exercises, see the Colab solutions page or the Streamlit page.
#

# %% [markdown]
# <img src="https://raw.githubusercontent.com/callummcdougall/Fundamentals/main/images/cnn.png" width="350">
#

# %% [markdown]
# # Introduction
#

# %% [markdown]
# This section is designed to get you familiar with basic neural networks: how they are structured, the basic operations like linear layers and convolutions which go into making them, and why they work as well as they do. You'll start by making very simple neural networks, and by the end of today you'll build up to assembling ResNet34, a comparatively much more complicated architecture.
#

# %% [markdown]
# ## Content & Learning Objectives
#

# %% [markdown]
# ### 1️⃣ Making your own modules
#
# In the first set of exercises, we'll cover the general structure of modules in PyTorch. You'll also implement your own basic modules, including for ReLU and Linear layers. You'll finish by assembling a very simple neural network.
#
# > ##### Learning objectives
# >
# > - Learn how to create your own modules in PyTorch, by inheriting from `nn.Module`
# > - Assemble the pieces together to create a simple fully-connected network, to classify MNIST digits
#
# ### 2️⃣ Training Neural Networks
#
# Here, you'll learn how to write a training loop in PyTorch. We'll keep it simple for today (and later on we'll experiment with more modular and extensible designs).
#
# > ##### Learning objectives
# >
# > - Understand how to work with transforms, datasets and dataloaders
# > - Understand the basic structure of a training loop
# > - Learn how to write your own validation loop
#
# ### 3️⃣ Convolutions
#
# In this section, you'll read about convolutions, and implement them as an `nn.Module` (not from scratch; we leave that to the bonus exercises). You'll also learn about maxpooling, and implement that as well.
#
# > ##### Learning Objectives
# >
# > - Learn how convolutions work, and why they are useful for vision models
# > - Implement your own convolutions, and maxpooling layers
#
# ### 4️⃣ ResNets
#
# Here, you'll combine all the pieces you've learned so far to assemble ResNet34, a much more complex architecture used for image classification.
#
# > ##### Learning Objectives
# >
# > - Learn about skip connections, and how they help overcome the degradation problem
# > - Learn about batch normalization, and why it is used in training
# > - Assemble your own ResNet, and load in weights from PyTorch's ResNet implementation
#
# ### 5️⃣ Bonus - Convolutions From Scratch
#
# This section takes you through the low-level details of how to actually implement convolutions. It's not necessary to understand this section to complete the exercises, but it's a good way to get a deeper understanding of how convolutions work.
#
# > ##### Learning objectives
# >
# > - Understand how array strides work, and why they're important for efficient linear operations
# > - Learn how to use `as_strided` to perform simple linear operations like trace and matrix multiplication
# > - Implement your own convolutions and maxpooling functions using stride-based methods
#
# ### 6️⃣ Bonus - Feature Extraction
#
# In this section, you'll learn how to repurpose your ResNet to perform a different task than it was designed for, using feature extraction.
#
# > ##### Learning Objectives
# >
# > - Understand the difference between feature extraction and finetuning
# > - Perform feature extraction on a pre-trained ResNet
#

# %% [markdown]
# ## Setup (don't read, just run!)
#

# %%

# Install packages
# %pip install einops
# %pip install jaxtyping


# Code to download the necessary files (e.g. solutions, test funcs)
# import os, sys
# if not os.path.exists("chapter0_fundamentals"):
#     !wget https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/main.zip
#     !unzip ./main.zip 'ARENA_3.0-main/chapter0_fundamentals/exercises/*'
#     os.remove("./main.zip")
#     os.rename("ARENA_3.0-main/chapter0_fundamentals", "chapter0_fundamentals")
#     os.rmdir("ARENA_3.0-main")
#     sys.path.insert(0, "chapter0_fundamentals/exercises")

# # Clear output
# from IPython.display import clear_output
# clear_output()
# print("Imports & installations complete!")


# %%
import functools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm

# Get file paths to this set of exercises
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"

if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import (
    imshow,
    line,
    bar,
    plot_train_loss_and_test_accuracy_from_trainer,
)
import part2_cnns.tests as tests
from part2_cnns.utils import print_param_count

# %% [markdown]
# <details>
# <summary>Help - I get a NumPy-related error</summary>
#
# This is an annoying colab-related issue which I haven't been able to find a satisfying fix for. If you restart runtime (but don't delete runtime), and run just the imports cell above again (but not the `%pip install` cell), the problem should go away.
#
# </details>
#

# %% [markdown]
# # 1️⃣ Making your own modules
#

# %% [markdown]
# > ### Learning objectives
# >
# > - Learn how to create your own modules in PyTorch, by inheriting from `nn.Module`
# > - Assemble the pieces together to create a simple fully-connected network, to classify MNIST digits
#

# %% [markdown]
# ## Subclassing `nn.Module`
#
# One of the most basic parts of PyTorch that you will see over and over is the `nn.Module` class. All types of neural net components inherit from it, from the simplest `nn.Relu` to the most complex `nn.Transformer`. Often, a complex `nn.Module` will have sub-`Module`s which implement smaller pieces of its functionality.
#
# Other common `Module`s you'll see include
#
# - `nn.Linear`, for fully-connected layers with or without a bias
# - `nn.Conv2d`, for a two-dimensional convolution (we'll see more of these in a future section)
# - `nn.Softmax`, which implements the [softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) function
#
# The list goes on, including activation functions, normalizations, pooling, attention, and more. You can see all the `Module`s that PyTorch provides [here](https://pytorch.org/docs/stable/nn.html). You can also create your own `Module`s, as we will do often!
#
# The `Module` class provides a lot of functionality, but we'll only cover a little bit of it here.
#
# In this section, we'll add another layer of abstraction to all the linear operations we've done in previous sections, by packaging them inside `nn.Module` objects.
#

# %% [markdown]
# ### `__init__` and `forward`
#
# A subclass of `nn.Module` usually looks something like this:
#
# ```python
# import torch.nn as nn
#
# class MyModule(nn.Module):
#     def __init__(self, arg1, arg2, ...):
#         super().__init__()
#         # Initialization code
#
#     def forward(self, x: t.Tensor) -> t.Tensor:
#         # Forward pass code
# ```
#
# The initialization sets up attributes that will be used for the life of the `Module`, like its parameters, hyperparameters, or other sub-`Module`s it might need to use. These are usually added to the instance with something like `self.attribute = attr`, where `attr` might be provided as an argument. Some modules are simple enough that they don't need any persistent attributes, and in this case you can skip the `__init__`.
#
# The `forward` method is called on each forward pass of the `Module`, possibly using the attributes that were set up in the `__init__`. It should take in the input, do whatever it's supposed to do, and return the result. Subclassing `nn.Module` automatically makes instances of your class callable, so you can do `model(x)` on an input `x` to invoke the `forward` method.
#

# %% [markdown]
# ### The `nn.Parameter` class
#
# A `nn.Parameter` is a special type of `Tensor`. Basically, this is the class that torch has provided for storing the weights and biases of a `Module`. It has some special properties for doing this:
#
# - If a `Parameter` is set as an attribute of a `Module`, it will be auto-detected by torch and returned when you call `module.parameters()` (along with all the other `Parameters` associated with the `Module`, or any of the `Module`'s sub-modules!).
# - This makes it easy to pass all the parameters of a model into an optimizer and update them all at once.
#
# When you create a `Module` that has weights or biases, be sure to wrap them in `nn.Parameter` so that torch can detect and update them appropriately:
#
# ```python
# class MyModule(nn.Module):
#     def __init__(self, weights: t.Tensor, biases: t.Tensor):
#         super().__init__()
#         self.weights = nn.Parameter(weights) # wrapping a tensor in nn.Parameter
#         self.biases = nn.Parameter(biases)
# ```
#

# %% [markdown]
# ### Printing information with `extra_repr`
#
# Another useful method is called `extra_repr`. This allows you to format the string representation of your `Module` in a way that's more informative than the default. For example, the following:
#
# ```python
# class MyModule(nn.Module):
#     def __init__(self, arg1, arg2, ...):
#         super().__init__()
#         # Initialization code
#
#     def extra_repr(self) -> str:
#         return f"arg1={self.arg1}, arg2={self.arg2}, ..."
# ```
#
# will result in the output `"MyModule(arg1=arg1, arg2=arg2, ...)"` when you print an instance of this module. You might want to take this opportunity to print out useful invariant information about the module. The Python built-in function `getattr` might be helpful here (it can be used e.g. as `getattr(self, "arg1")`, which returns the same as `self.arg1` would). For simple modules, it's fine not to implement `extra_repr`.
#

# %% [markdown]
# ## ReLU
#
# The first module you should implement is `ReLU`. This will relatively simple, since it doesn't involve any argument (so we only need to implement `forward`). Make sure you look at the PyTorch documentation page for [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) so that you're comfortable with what they do and why they're useful in neural networks.
#

# %% [markdown]
# ### Exercise - implement `ReLU`
#
# ```yaml
# Difficulty: 🔴🔴⚪⚪⚪
# Importance: 🔵🔵🔵⚪⚪
#
# You should spend up to ~10 minutes on this exercise.
# ```
#
# You should fill in the `forward` method of the `ReLU` class below.
#

# %%
# x = t.randn((2, 3, 4))
# x, t.maximum(t.zeros_like(x), x)


# %%
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(t.zeros_like(x), x)


# tests.test_relu(ReLU)

# %% [markdown]
# ## Linear
#
# Now implement your own `Linear` module. This applies a simple linear transformation, with a weight matrix and optional bias vector. The PyTorch documentation page is [here](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html). Note that this is the first `Module` you'll implement that has learnable weights and biases.
#
# <details>
# <summary>Question - what type should these variables be?</summary>
#
# They have to be `torch.Tensor` objects wrapped in `nn.Parameter` in order for `nn.Module` to recognize them. If you forget to do this, `module.parameters()` won't include your `Parameter`, which prevents an optimizer from being able to modify it during training.
#
# Also, in tomorrow's exercises we'll be building a ResNet and loading in weights from a pretrained model, and this is hard to do if you haven't registered all your parameters!
#
# </details>
#
# For any layer, initialization is very important for the stability of training: with a bad initialization, your model will take much longer to converge or may completely fail to learn anything. The default PyTorch behavior isn't necessarily optimal and you can often improve performance by using something more custom, but we'll follow it for today because it's simple and works decently well.
#
# Each float in the weight and bias tensors are drawn independently from the uniform distribution on the interval:
#
# $$
# \bigg[-\frac{1}{\sqrt{N_{in}}}, \frac{1}{\sqrt{N_{in}}}\bigg]
# $$
#
# where $N_{in}$ is the number of inputs contributing to each output value. The rough intuition for this is that it keeps the variance of the activations at each layer constant, since each one is calculated by taking the sum over $N_{in}$ inputs multiplied by the weights (and standard deviation of the sum of independent random variables scales as the square root of number of variables).
#
# The name for this is **Kaiming (uniform) initialisation**.
#
# <details>
# <summary>Technical details (derivation of distribution)</summary>
#
# The key intuition behind Kaiming initialisation (and others like it) is that we want the variance of our activations to be the same through all layers of the model when we initialize. Suppose $x$ and $y$ are activations from two adjacent layers, and $w$ are the weights connecting them (so we have $y_i = \sum_j w_{ij} x_j + b_i$, where $b$ is the bias). With $N_{x}$ as the number of neurons in layer $x$, we have:
#
# $$
# \begin{aligned}
# \operatorname{Var}\left(y_i\right)=\sigma_x^2 & =\operatorname{Var}\left(\sum_j w_{i j} x_j\right) \\
# & =\sum_j \operatorname{Var}\left(w_{i j} x_j\right) \quad \text { Inputs and weights are independent of each other } \\
# & =\sum_j \operatorname{Var}\left(w_{i j}\right) \cdot \operatorname{Var}\left(x_j\right) \quad \text { Variance of product of independent RVs with zero mean is product of variances } \\
# & = N_x \cdot \sigma_x^2 \cdot \operatorname{Var}\left(w_{i j}\right) \quad \text { Variance equal for all } N_x \text { neurons, call this value } \sigma_x^2
# \end{aligned}
# $$
#
# For this to be the same as $\sigma_x^2$, we need $\operatorname{Var}(w_{ij}) = \frac{1}{N_x}$, so the standard deviation is $\frac{1}{\sqrt{N_x}}$.
#
# This is not exactly the case for the Kaiming uniform distribution (which has variance $\frac{12}{(2 \sqrt{N_x})^2} = \frac{3}{N_x}$), and as far as I'm aware there's no principled reason why PyTorch does this. But the most important thing is that the variance scales as $O(1 / N_x)$, rather than what the exact scaling constant is.
#
# There are other initializations with some theoretical justification. For instance, **Xavier initialization** has a uniform distribution in the interval:
#
# $$
# \bigg[-\frac{\sqrt{6}}{\sqrt{N_{in} + N_{out} + 1}}, \frac{\sqrt{6}}{\sqrt{N_{in} + N_{out} + 1}}\bigg]
# $$
#
# which is motivated by the idea of both keeping the variance of activations constant and keeping the **_gradients_** constant when we backpropagate.
#
# However, you don't need to worry about any of this here, just implement Kaiming He uniform with a bound of $\frac{1}{\sqrt{N_{in}}}$!
#
# </details>
#

# %% [markdown]
# ### Exercise - implement `Linear`
#
# ```yaml
# Difficulty: 🔴🔴⚪⚪⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to ~10 minutes on this exercise.
# ```
#
# Remember, you should define the weights (and bias, if appropriate) in the `__init__` block. Also, make sure not to mix up `bias` (which is the boolean parameter to `__init__`) and `self.bias` (which should either be the actual bias tensor, or `None` if `bias` is false).
#


# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        scaling_factor = 1 / np.sqrt(in_features)
        self.weight = nn.Parameter(
            scaling_factor * (2 * t.rand(out_features, in_features) - 1)
        )

        if bias:
            self.bias = nn.Parameter(scaling_factor * (2 * t.rand(out_features) - 1))
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        # out = t.matmul(x, self.weight.transpose(0, 1))
        out = einops.einsum(
            x,
            self.weight,
            "... in_features, out_features in_features -> ... out_features",
        )
        if self.bias is not None:
            out += self.bias

        return out

    def extra_repr(self) -> str:
        return f"Linear module with weights: {self.weight.shape} and bias: {self.bias.shape if self.bias is not None else None}"


# tests.test_linear_parameters(Linear, bias=False)
# tests.test_linear_parameters(Linear, bias=True)
# tests.test_linear_forward(Linear, bias=False)
# tests.test_linear_forward(Linear, bias=True)

# %%


# %% [markdown]
# <details>
# <summary>Help - when I print my Linear module, it also prints a large tensor.</summary>
#
# This is because you've (correctly) defined `self.bias` as either `torch.Tensor` or `None`, rather than set it to the boolean value of `bias` used in initialisation.
#
# To fix this, you will need to change `extra_repr` so that it prints the boolean value of `bias` rather than the value of `self.bias`.
#
# </details>
#

# %% [markdown]
# ## Flatten
#
# Lastly, we'll implement `Flatten`. This is a standardised way to rearrange our tensors so that they can be fed into a linear layer. It's a bit like `einops.flatten`, but more specialised (we recommend you use the torch `reshape` method rather than `einops` for this exercise, although it is possible to use einops).
#

# %% [markdown]
#

# %% [markdown]
# ### Exercise - implement `Flatten`
#
# ```c
# Difficulty: 🔴🔴🔴🔴⚪
# Importance: 🔵🔵🔵⚪⚪
#
# You should spend up to 10-15 minutes on this exercise.
# ```
#

# %%


# %%
# m = nn.Flatten(1, 2)

# x = t.randint(100, (2, 3, 4))
# x, m(x), t.reshape(x, [2, 1, 12])


# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        dim_sizes = input.shape
        dims = len(dim_sizes)
        start_idx = self.start_dim if self.start_dim >= 0 else (dims + self.start_dim)
        end_idx = self.end_dim if self.end_dim >= 0 else (dims + self.end_dim)

        if end_idx < start_idx:
            raise RuntimeError(
                f"End cannot be before start. start: {self.start_dim}, end: {self.end_dim}"
            )

        if end_idx == start_idx:
            return input

        # get the sizes from start dim to end dim
        new_shape = []
        merged_size = 1
        for dim_idx in range(0, dims):
            if dim_idx >= start_idx and dim_idx <= end_idx:
                merged_size *= dim_sizes[dim_idx]
                if dim_idx == end_idx:
                    new_shape.append(merged_size)
            else:
                new_shape.append(dim_sizes[dim_idx])

        # if still empty, means we probably had 0, -1
        if len(new_shape) == 0:
            new_shape = [merged_size]

        # print(input.shape, self.start_dim, self.end_dim, start_idx, end_idx, merged_size, new_shape)

        return t.reshape(input, new_shape)

    def extra_repr(self) -> str:
        pass


# tests.test_flatten(Flatten)

# %% [markdown]
# <details>
# <summary>Help - I can't figure out what shape the output should be in Flatten.</summary>
#
# If `input.shape = (n0, n1, ..., nk)`, and the `Flatten` module has `start_dim=i, end_dim=j`, then the new shape should be `(n0, n1, ..., ni*...*nj, ..., nk)`. This is because we're **flattening** over the dimensions `(ni, ..., nj)`.
#
# </details>
#
# <details>
# <summary>Help - I can't see why my Flatten module is failing the tests.</summary>
#
# The most common reason is failing to correctly handle indices. Make sure that:
#
# - You're indexing up to **and including** `end_dim`.
# - You're correctly managing the times when `end_dim` is negative (e.g. if `input` is an nD tensor, and `end_dim=-1`, this should be interpreted as `end_dim=n-1`).
# </details>
#

# %% [markdown]
# ## Simple Multi-Layer Perceptron
#
# Now, we can put together these two modules to create a neural network. We'll create one of the simplest networks which can be used to separate data that is non-linearly separable: a single linear layer, followed by a nonlinear function (ReLU), followed by another linear layer. This type of architecture (alternating linear layers and nonlinear functions) is often called a **multi-layer perceptron** (MLP).
#
# The output of this network will have 10 dimensions, corresponding to the 10 classes of MNIST digits. We can then use the **softmax function** $x_i \to \frac{e^{x_i}}{\sum_i e^{x_i}}$ to turn these values into probabilities. However, it's common practice for the output of a neural network to be the values before we take softmax, rather than after. We call these pre-softmax values the **logits**.
#
# <details>
# <summary>Question - can you see what makes logits non-unique (i.e. why any given set of probabilities might correspond to several different possible sets of logits)?</summary>
#
# Logits are **translation invariant**. If you add some constant $c$ to all logits $x_i$, then the new probabilities are:
#
# $$
# p_i' = \frac{e^{x_i + c}}{\sum_j e^{x_j + c}} = \frac{e^{x_i}}{\sum_j e^{x_j}} = p_i
# $$
#
# in other words, the probabilities don't change.
#
# We can define **logprobs** as the log of the probabilities, i.e. $y_i = \log p_i$. Unlike logits, these are uniquely defined.
#
# </details>
#

# %% [markdown]
# ### Exercise - implement the simple MLP
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to ~20 minutes on this exercise.
# ```
#
# The diagram below shows what your MLP should look like:
#
# <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/mlp-mermaid.svg" width="170">
#
# Please ask a TA (or message the Slack group) if any part of this diagram is unclear.
#


# %%
class SimpleMLP(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.flatten = Flatten()
        self.linear1 = Linear(28**2, 100)
        self.relu = ReLU()
        self.linear2 = Linear(100, 10)
        self.debug = debug

    def forward(self, x: t.Tensor) -> t.Tensor:

        if self.debug:
            print(f"Initial shape: {x.shape}")
        hidden_states = self.flatten(x)

        if self.debug:
            print(f"After flatten shape: {hidden_states.shape}")
        hidden_states = self.linear1(hidden_states)

        if self.debug:
            print(f"After linear1 val: {hidden_states}")
        hidden_states = self.relu(hidden_states)

        if self.debug:
            print(f"After relu val: {hidden_states}")
        hidden_states = self.linear2(hidden_states)

        if self.debug:
            print(f"After linear2 vals: {hidden_states}")

        return hidden_states


# tests.test_mlp_module(SimpleMLP)
# tests.test_mlp_forward(SimpleMLP)

# %%


# %%


# %% [markdown]
# In the next section, we'll learn how to train and evaluate our model on real data.
#

# %% [markdown]
# # 2️⃣ Training Neural Networks
#

# %% [markdown]
# ## Transforms, Datasets & DataLoaders
#
# Before we use this model to make any predictions, we first need to think about our input data. Below is a block of code to fetch and process MNIST data. We will go through it line by line.
#

# %%
# MNIST_TRANSFORM = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
# )


# def get_mnist(subset: int = 1):
#     """Returns MNIST training data, sampled by the frequency given in `subset`."""
#     mnist_trainset = datasets.MNIST(
#         root="./data", train=True, download=True, transform=MNIST_TRANSFORM
#     )
#     mnist_testset = datasets.MNIST(
#         root="./data", train=False, download=True, transform=MNIST_TRANSFORM
#     )

#     if subset > 1:
#         mnist_trainset = Subset(
#             mnist_trainset, indices=range(0, len(mnist_trainset), subset)
#         )
#         mnist_testset = Subset(
#             mnist_testset, indices=range(0, len(mnist_testset), subset)
#         )

#     return mnist_trainset, mnist_testset

# %%


# %% [markdown]
# The `torchvision` package consists of popular datasets, model architectures, and common image transformations for computer vision. `transforms` is a library from `torchvision` which provides access to a suite of functions for preprocessing data.
#
# We define a transform for the MNIST data (which is applied to each image in the dataset) by composing `ToTensor` (which converts a `PIL.Image` object into a PyTorch tensor) and `Normalize` (which takes arguments for the mean and standard deviation, and performs the linear transformation `x -> (x - mean) / std`).
#

# %% [markdown]
# Next, we define our datasets, using the `torchvision.datasets` library. The argument `root="./data"` indicates that we're storing our data in the `./data` directory, and `transform=MNIST_TRANSFORM` tells us that we should apply our previously defined `transform` to each element in our dataset.
#
# The `Subset` function allows us to take a subset of a dataset. The argument `indices` is a list of indices to sample from the dataset. For example, `Subset(mnist_trainset, indices=[0, 1, 2])` will return a dataset containing only the first three elements of `mnist_trainset`.
#

# %% [markdown]
# Finally, `DataLoader` provides a useful abstraction to work with a dataset. It takes in a dataset, and a few arguments including `batch_size` (how many inputs to feed through the model on which to compute the loss before each step of gradient descent) and `shuffle` (whether to randomise the order each time you iterate). The object that it returns can be iterated through as follows:
#
# ```python
# for X, y in mnist_trainloader:
#     ...
# ```
#
# where `X` is a 3D array of shape `(batch_size, 28, 28)` where each slice is an image, and `y` is a 1D tensor of labels of length `batch_size`. Without using this helpful object, we'd have to iterate through our dataset as follows:
#
# ```python
# for i in range(len(mnist_trainset) // batch_size):
#
#     X = mnist_trainset.data[i*batch_size: (i+1)*batch_size]
#     y = mnist_trainset.targets[i*batch_size: (i+1)*batch_size]
#
#     ...
# ```
#
# A note about batch size - it's common to see batch sizes which are powers of two. The motivation is for efficient GPU utilisation, since processor architectures are normally organised around powers of 2, and computational efficiency is often increased by having the items in each batch split across processors. Or at least, that's the idea. The truth is a bit more complicated, and some studies dispute whether it actually saves time. We'll dive much deeper into these kinds of topics during the week on training at scale.
#

# %% [markdown]
# ---
#

# %% [markdown]
# Before proceeding, try and answer the following questions:
#
# <details>
# <summary>Question - can you explain why we include a data normalization function in <code>torchvision.transforms</code> ?</summary>
#
# One consequence of unnormalized data is that you might find yourself stuck in a very flat region of the domain, and gradient descent may take much longer to converge.
#
# Normalization isn't strictly necessary for this reason, because any rescaling of an input vector can be effectively undone by the network learning different weights and biases. But in practice, it does usually help speed up convergence.
#
# Normalization also helps avoid numerical issues.
#
# </details>
#
# <details>
# <summary>Question - what is the benefit of using <code>shuffle=True</code> when defining our dataloaders? What might the problem be if we didn't do this?</summary>
#
# Shuffling is done during the training to make sure we aren't exposing our model to the same cycle (order) of data in every epoch. It is basically done to ensure the model isn't adapting its learning to any kind of spurious pattern.
#
# </details>
#

# %% [markdown]
# ---
#

# %% [markdown]
# ### Aside - `tqdm`
#
# You might have seen some blue progress bars running when you first downloaded your MNIST data. These were generated using a library called `tqdm`, which is also a really useful tool when training models or running any process that takes a long period of time.
#
# You can run the cell below to see how these progress bars are used (note that you might need to install the `tqdm` library first).
#

# %%
# from tqdm.notebook import tqdm
# import time

# for i in tqdm(range(100)):
#     time.sleep(0.01)

# %%


# %% [markdown]
# `tqdm` wraps around a list, range or other iterable, but other than that it doesn't affect the structure of your loop.
#
# One gotcha when it comes to `tqdm` - you need to make sure you pass it something with a well-defined length. For instance, if you pass it an `enumerate` or `zip` object, it won't work as expected because it can't infer length from the object. You can fix this problem by wrapping your iterator in a list (e.g. `tqdm(list(zip(...)))`).
#

# %% [markdown]
# ### Aside - `device`
#
# One last thing to discuss before we move onto training our model: **GPUs**. We'll discuss this in more detail in later exercises. For now, [this page](https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk) should provide a basic overview of how to use your GPU. A few things to be aware of here:
#
# - The `to` method is really useful here - it can move objects between different devices (i.e. CPU and GPU) _as well as_ changing a tensor's datatype.
#   - Note that `to` is never inplace for tensors (i.e. you have to call `x = x.to(device)`), but when working with models, calling `model = model.to(device)` or `model.to(device)` are both perfectly valid.
# - Errors from having one tensor on cpu and another on cuda are very common. Some useful practices to avoid this:
#   - Throw in assert statements, to make sure tensors are on the same device
#   - Remember that when you initialise an array (e.g. with `t.zeros` or `t.arange`), it will be on CPU by default.
#   - Tensor methods like [`new_zeros`](https://pytorch.org/docs/stable/generated/torch.Tensor.new_zeros.html) or [`new_full`](https://pytorch.org/docs/stable/generated/torch.Tensor.new_full.html) are useful, because they'll create tensors which match the device and dtype of the base tensor.
#
# It's common practice to put a line like this at the top of your file, defining a global variable which you can use in subsequent modules and functions (excluding the print statement):
#

# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
# print(device)

# %% [markdown]
# ## Training loop
#
# Below is a very simple training loop, which you can run to train your model.
#
# In later exercises, we'll try to **modularize** our training loops. This will involve things like creating a `Trainer` class which wraps around our model, and giving it methods like `training_step` and `validation_step` which correspond to different parts of the training loop. This will make it easier to add features like logging and validation, and will also make our code more readable and easier to refactor. However, for now we've kept things simple.
#

# %%
# try:
#     del model
# except NameError:
#     pass

# model = SimpleMLP().to(device)

# batch_size = 64
# epochs = 3

# mnist_trainset, mnist_testset = get_mnist(subset=3)
# mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
# mnist_testloader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)

# optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
# loss_list = []

# for epoch in tqdm(range(epochs)):
#     for imgs, labels in mnist_trainloader:
#         imgs = imgs.to(device)
#         labels = labels.to(device)
#         logits = model(imgs)
#         loss = F.cross_entropy(logits, labels)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         loss_list.append(loss.item())

# line(
#     loss_list,
#     yaxis_range=[0, max(loss_list) + 0.1],
#     labels={"x": "Num batches seen", "y": "Cross entropy loss"},
#     title="SimpleMLP training on MNIST",
#     width=700,
# )

# %% [markdown]
# Let's break down the important parts of this code.
#
# The batch size is the number of samples in each batch (i.e. the number of samples we feed into the model at once). While training our model, we differentiate with respect to the average loss over all samples in the batch (so a smaller batch usually means the loss is more noisy). However, if you're working with large models, then often having a batch size too large will result in a memory error. This will be relevant for models later on in the course, but for now we're working with very small models so this isn't an issue.
#
# Next, we get our training set, via the helper function `get_mnist`. This helper function used `torchvision.datasets.MNIST` to load in data, and then (optionally) the `torch.utils.data.Subset` function to return a subset of this data. Don't worry about the details of this function, it's not the kind of thing you'll need to know by heart.
#
# We then define our optimizer, using `torch.optim.Adam`. The `torch.optim` module gives a wide variety of modules, such as Adam, SGD, and RMSProp. Adam is generally the most popular and seen as the most effective in the majority of cases. We'll discuss optimizers in more detail tomorrow, but for now it's enough to understand that the optimizer calculates the amount to update parameters by (as a function of those parameters' gradients, and sometimes other inputs), and performs this update step. The first argument passed to our optimizer is the parameters of our model (because these are the values that will be updated via gradient descent), and you can also pass keyword arguments to the optimizer which change its behaviour (e.g. the learning rate).
#
# Lastly, we have the actual training loop. We iterate through our training data, and for each batch we:
#
# 1. Evaluate our model on the batch of data, to get the logits for our class predictions,
# 2. Calculate the loss between our logits and the true class labels,
# 3. Backpropagate the loss through our model (this step accumulates gradients in our model parameters),
# 4. Step our optimizer, which is what actually updates the model parameters,
# 5. Zero the gradients of our optimizer, ready for the next step.
#

# %% [markdown]
# ### Cross entropy loss
#
# The formula for cross entropy loss over a batch of size $N$ is:
#
# $$
# \begin{aligned}
# l &= \frac{1}{N} \sum_{n=1}^{N} l_n \\
# l_n &=-\log p_{n, y_{n}}
# \end{aligned}
# $$
#
# where $p_{n, c}$ is the probability the model assigns to class $c$ for sample $n$, and $y_{n}$ is the true label for this sample.
#
# <details>
# <summary>See this dropdown, if you're still confused about this formula, and how this relates to the information-theoretic general formula for cross entropy.</summary>
#
# The cross entropy of a distribution $p$ relate to a distribution $q$ is:
#
# $$
# \begin{aligned}
# H(q, p) &= -\sum_{n} q(n) \log p(n)
# \end{aligned}
# $$
#
# In our case, $q$ is the true distribution (i.e. the one-hot encoded labels, which equals one for $n = y_n$, zero otherwise), and $p$ is our model's output. With these subsitutions, this formula becomes equivalent to the formula for $l$ given above.
#
# </details>
#
# <details>
# <summary>See this dropdown, if you're confused about how this is the same as the <a href="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss">PyTorch definition</a>.</summary>
#
# The PyTorch definition of cross entropy loss is:
#
# $$
# \ell(x, y)=\frac{1}{N}\sum_{n=1}^{N} l_n, \quad l_n=-\sum_{c=1}^C w_c \log \frac{\exp \left(x_{n, c}\right)}{\sum_{i=1}^C \exp \left(x_{n, i}\right)} y_{n, c}
# $$
#
# $w_c$ are the weights (which all equal one by default), $p_{n, c} = \frac{\exp \left(x_{n, c}\right)}{\sum_{i=1}^C \exp \left(x_{n, i}\right)}$ are the probabilities, and $y_{n, c}$ are the true labels (which are one-hot encoded, i.e. their value is one at the correct label $c$ and zero everywhere else). With this, the formula for $l_n$ reduces to the one we see above (i.e. the mean of the negative log probabilities).
#
# </details>
#
# The function `torch.functional.cross_entropy` expects the **unnormalized logits** as its first input, rather than probabilities. We get probabilities from logits by applying the softmax function:
#
# $$
# \begin{aligned}
# p_{n, c} &= \frac{\exp(x_{n, c})}{\sum_{c'=1}^{C} \exp(x_{n, c'})}
# \end{aligned}
# $$
#
# where $x_{n, c}$ is the model's output for class $c$ and sample $n$, and $C$ is the number of classes (in the case of MNIST, $C = 10$).
#
# Some terminology notes:
#
# - When we say **logits**, we mean the output of the model before applying softmax. We can uniquely define a distribution with a set of logits, just like we can define a distribution with a set of probabilities (and sometimes it's easier to think of a distribution in terms of logits, as we'll see later in the course).
#
# - When we say **unnormalized**, we mean the denominator term $\sum_{c'} \exp(x_{n, c'})$ isn't necessarily equal to 1. We can add a constant value onto all the logits which makes this term 1 without changing any of the actual probabilities, then we have the relation $p_{n, c} = \exp(-l_{n, c})$. Here, we call $-l_{n, c}$ the **log probabilities** (or log probs), since $-l_{n, c} = \log p_{n, c}$.
#
# If you're interested in the intuition behind cross entropy as a loss function, see [this post on KL divergence](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence) (note that KL divergence and cross entropy differ by an amount which is independent of our model's predictions, so minimizing cross entropy is equivalent to minimizing KL divergence). Also see these two videos:
#
# - [Intuitively Understanding the Cross Entropy Loss](https://www.youtube.com/watch?v=Pwgpl9mKars&ab_channel=AdianLiusie)
# - [Intuitively Understanding the KL Divergence](https://www.youtube.com/watch?v=SxGYPqCgJWM&ab_channel=AdianLiusie)
#

# %% [markdown]
# ### Aside - `dataclasses`
#
# Sometimes, when we have a lot of different input parameters to our model, it can be helpful to use dataclasses to keep track of them all. Dataclasses are a special kind of class which come with built-in methods for initialising and printing (i.e. no need to define an `__init__` or `__repr__`). Another advantage of using them is autocompletion: when you type in `args.` in VSCode, you'll get a dropdown of all your different dataclass attributes, which can be useful when you've forgotten what you called a variable!
#
# Here's an example of how we might rewrite our training code above using dataclasses. We've wrapped all the training code inside a single argument called `train`, which takes a `SimpleMLPTrainingArgs` object as its only argument.
#


# %%
@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = SimpleMLPTrainingArgs(batch_size=128).
    """

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    subset: int = 2


def train(args: SimpleMLPTrainingArgs):
    """
    Trains the model, using training parameters from the `args` object.
    """
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist(subset=args.subset)
    mnist_trainloader = DataLoader(
        mnist_trainset, batch_size=args.batch_size, shuffle=True
    )
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []

    for epoch in tqdm(range(args.epochs)):
        for imgs, labels in mnist_trainloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())

        # validate after each epoch
        with t.inference_mode():
            total = 0
            correct = 0
            for imgs, labels in mnist_testloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                correct += (t.argmax(logits, dim=-1) == labels).sum().item()
                total += args.batch_size

            print(f"Accuracy: {correct * 100.0 / total}%")

    line(
        loss_list,
        yaxis_range=[0, max(loss_list)],
        labels={"x": "Num batches seen", "y": "Cross entropy loss"},
        title="SimpleMLP training on MNIST",
        width=700,
    )


# args = SimpleMLPTrainingArgs()
# train(args)

# %% [markdown]
# ### Exercise - add a validation loop
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵🔵🔵🔵
#
# You should spend up to ~20 minutes on this exercise.
#
# It is very important that you understand training loops and how they work, because we'll be doing a lot of model training in this way.
# ```
#
# Edit the `train` function above to include a validation loop. Train your model, making sure you measure the accuracy at the end of each epoch.
#
# Here are a few tips to help you:
#
# - During the validation step, you should be measuring **accuracy**, which is defined as **the fraction of correctly classified images**.
#   - Note that (unlike loss) accuracy should only be logged after you've gone through the whole validation set. This is because your model doesn't update between computing different accuracies, so it doesn't make sense to log all of them separately.
# - You don't need to convert to probabilities before calculating accuracy (or even to logprobs), because softmax is an order-preserving function.
# - You can wrap your code in `with t.inference_mode():` to make sure that your model is in inference mode during validation (i.e. gradients don't propagate).
#   - Note you could also use the decorator `@t.inference_mode()` to do this, if your training loop was a function.
# - The `get_mnist` function returns both a trainset and a testset. In the `train` function above we only used the first one, but you should use both.
# - You'll need a dataloader for the testset, just like we did for the trainset. It doesn't matter whether you shuffle the testset or not, because we're not updating our model parameters during validation.
#

# %%
# YOUR CODE HERE - add a validation loop


@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = SimpleMLPTrainingArgs(batch_size=128).
    """

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    subset: int = 5


def train(args: SimpleMLPTrainingArgs):
    """
    Trains the model, using training parameters from the `args` object.
    """
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist(subset=args.subset)
    mnist_trainloader = DataLoader(
        mnist_trainset, batch_size=args.batch_size, shuffle=True
    )
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []

    for epoch in tqdm(range(args.epochs)):
        for imgs, labels in mnist_trainloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())

        # validate after each epoch
        with t.inference_mode():
            total = 0
            correct = 0
            for imgs, labels in mnist_testloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                correct += (t.argmax(logits, dim=-1) == labels).sum().item()
                total += args.batch_size

            print(f"Accuracy: {correct * 100.0 / total}%")

    line(
        loss_list,
        yaxis_range=[0, max(loss_list) + 0.1],
        labels={"x": "Num batches seen", "y": "Cross entropy loss"},
        title="SimpleMLP training on MNIST",
        width=700,
    )


# args = SimpleMLPTrainingArgs()
# train(args)

# %% [markdown]
# <details>
# <summary>Help - I'm not sure how to measure correct classifications.</summary>
#
# You can take argmax of the output of your model, using `torch.argmax` (with the keyword argument `dim` to specify the dimension you want to take max over).
#
# </details>
#
# <details>
# <summary>Help - I get <code>RuntimeError: expected scalar type Float but found Byte</code>.</summary>
#
# This is commonly because one of your operations is between tensors with the wrong datatypes (e.g. `int` and `float`). Try navigating to the error line and checking your dtypes (or using VSCode's built-in debugger).
#
# </details>
#

# %% [markdown]
# You should find that after the first epoch, the model is already doing much better than random chance, and it improves slightly in subsequent epochs.
#

# %% [markdown]
# # 3️⃣ Convolutions
#

# %% [markdown]
# > ### Learning Objectives
# >
# > - Learn how convolutions work, and why they are useful for vision models
# > - Implement your own convolutions, and maxpooling layers
#

# %% [markdown]
# ## Reading
#
# - [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) by 3Blue1Brown
# - [A Comprehensive Guide to Convolutional Neural Networks (TowardsDataScience)](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
#

# %% [markdown]
# ## Convolutions
#
# Here are some questions about convolutions to make sure you've understood the material. Once you finish the article above, you should try and answer these questions without referring back to the original article.
#
# <details>
# <summary>Why would convolutional layers be less likely to overfit data than standard linear (fully connected) layers?</summary>
#
# Convolutional layers require significantly fewer weights to be learned. This is because the same kernel is applied all across the image, rather than every pair of `(input, output)` nodes requiring a different weight to be learned.
#
# </details>
#
# <details>
# <summary>Suppose you fixed some random permutation of the pixels in an image, and applied this to all images in your dataset, before training a convolutional neural network for classifying images. Do you expect this to be less effective, or equally effective?</summary>
#
# It will be less effective, because CNNs work thanks to **spatial locality** - groups of pixels close together are more meaningful. For instance, CNNs will often learn convolutions at an early layer which recognise gradients or simple shapes. If you permute the pixels (even if you permute in the same way for every image), you destroy locality.
#
# </details>
#
# <details>
# <summary>If you have a 28x28 image, and you apply a 3x3 convolution with stride 1, padding 1, what shape will the output be?</summary>
#
# It will be the same shape, i.e. `28x28`. In the post linked above, this is described as **same padding**. Tomorrow, we'll build an MNIST classifier which uses these convolutions.
#
# </details>
#
# <br>
#

# %% [markdown]
# A note on terminology - you might see docs and docstrings sometimes use `num_features`, sometimes use `channels`, and sometimes just `C`. Often these two terms interchangeable. Our neural network inputs will often be RGB images and so we will have `channels=3` corresponding to the 3 colors red/green/blue. As we pass our initial image through convolutional layers, the number of channels will change. In the context of convolutions, the number of features and number of channels usually refer to the same value.
#

# %% [markdown]
# ### Exercise - implement `Conv2d`
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to ~20 minutes on this exercise.
#
# Make sure you understand what operation is taking place here, and how the dimensions are changing.
# ```
#
# Rather than implementing the `conv2d` function from scratch, we'll allow you to use `t.nn.functional.conv2d`. In the exercise below, you should use this function to implement the `nn.Conv2d` layer. In other words, you should:
#
# - Initialize the weights for the convolutional layer in the `__init__` function.
#   - You should look at the PyTorch page for `nn.Conv2d` [here](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) to understand what the shape of the weights should be.
#   - We assume `bias=False`, so the only `nn.Parameter` object we need to define is `weight`.
#   - You should use **uniform Xavier initialization**, which is described at the bottom of the `nn.Conv2d` docs (the bullet points under the **Variables** header).
# - Implement the `forward` method, which should apply the convolutional layer to the input.
#   - In other words, it should implement the `torch.nn.functional.conv2d` function (documentation [here](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html)), using the weights and biases (and other layer parameters) that you initialized in `__init__`.
# - Fill in the `extra_repr` method, to print out the convolutional layer's parameters.
#

# %%
from torch.nn.functional import conv2d


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        debug=False,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        super().__init__()
        # input is in_channels, h, w
        # the kernel should be... what?
        # out_channels, in_channels, kernel_size, kernel_size
        # scaling_factor for xavier initialization
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        scaling_factor = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(
            scaling_factor
            * (2 * t.rand(out_channels, in_channels, kernel_size, kernel_size) - 1)
        )

        self.stride = stride
        self.padding = padding
        self.debug = debug

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d, which you can import."""
        out = conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        if self.debug:
            print(f"Out shape: {out.shape}")

        return out

    def extra_repr(self) -> str:
        return (
            f"In channels: {self.in_channels}\n"
            + f"Out channels: {self.out_channels}\n"
            + f"Kernel size: {self.kernel_size}\n"
            + f"Weights shape: {self.weight.shape}."
        )


# tests.test_conv2d_module(Conv2d)
# m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
# print(f"Manually verify that this is an informative repr: {m}")

# %% [markdown]
# <details>
# <summary>Help - I don't know what to use as number of inputs, when doing Xavier initialisation.</summary>
#
# In the case of convolutions, each value in the output is computed by taking the product over `in_channels * kernel_width * kernel_height` elements. So this should be our value for $N_{in}$.
#
# </details>
#

# %% [markdown]
# ### Exercise - implement `MaxPool2d`
#
# ```yaml
# Difficulty: 🔴🔴⚪⚪⚪
# Importance: 🔵🔵⚪⚪⚪
#
# You should spend up to ~10 minutes on this exercise.
# ```
#
# Next, you should implement `MaxPool2d`. This module is often applied after a convolutional layer, to reduce the spatial dimensions of the output. It works by taking the maximum value in each kernel-sized window, and outputting that value. For instance, if we have a 2x2 kernel, then we take the maximum of each 2x2 window in the input.
#
# You should use `torch.nn.functional.max_pool2d` to implement this layer. The documentation page can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html), and the documentation for `nn.MaxPool2d` can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html).
#

# %%
from torch.nn.functional import max_pool2d


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Call the functional version of max_pool2d."""
        return max_pool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return (
            f"Kernel size: {self.kernel_size}\n"
            + f"Stride: {self.stride}\n"
            + f"Padding: {self.padding}"
        )


# tests.test_maxpool2d_module(MaxPool2d)
# m = MaxPool2d(kernel_size=3, stride=2, padding=1)
# print(f"Manually verify that this is an informative repr: {m}")

# %% [markdown]
# <details>
# <summary>Help - I'm really confused about what to do here!</summary>
#
# Your `forward` method should just implement the `maxpool2d` function. In order to get the parameters for this function like `kernel_size` and `stride`, you'll need to initialise them in `__init__`.
#
# Remember that `MaxPool2d` has no weights - it's just a wrapper for the `maxpool2d` function.
#
# ---
#
# Ideally, the `extra_repr` method should output something like:
#
# ```python
# "kernel_size=3, stride=2, padding=1"
# ```
#
# so that when you print the module, it will look like this:
#
# ```python
# MaxPool2d(kernel_size=3, stride=2, padding=1)
# ```
#
# </details>
#

# %% [markdown]
# # 4️⃣ ResNets
#

# %% [markdown]
# > ### Learning Objectives
# >
# > - Learn about skip connections, and how they help overcome the degradation problem
# > - Learn about batch normalization, and why it is used in training
# > - Assemble your own ResNet, and load in weights from PyTorch's ResNet implementation
#

# %% [markdown]
# ## Reading
#
# - [Batch Normalization in Convolutional Neural Networks](https://www.baeldung.com/cs/batch-normalization-cnn)
# - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
#
# You should move on once you can answer the following questions:
#
# <details>
# <summary>"Batch Normalization allows us to be less careful about initialization." Explain this statement.</summary>
#
# Weight initialisation methods like Xavier (which we encountered yesterday) are based on the idea of making sure the activations have approximately the same distribution across layers at initialisation. But batch normalisation ensures that this is the case as signals pass through the network.
#
# </details>
#
# <details>
# <summary>Give three reasons why batch norm improves the performance of neural networks.</summary>
#
# The reasons given in the first linked document above are:
#
# - Normalising inputs speeds up computation
# - Internal covariate shift is reduced, i.e. the mean and standard deviation is kept constant across the layers.
# - Regularisation effect: noise internal to each minibatch is reduced
# </details>
#
# <details>
# <summary>If you have an input tensor of size (batch, channels, width, height), and you apply a batchnorm layer, how many learned parameters will there be?</summary>
#
# A mean and standard deviation is calculated for each channel (i.e. each calculation is done across the batch, width, and height dimensions). So the number of learned params will be `2 * channels`.
#
# </details>
#
# <details>
# <summary>In the paper, the diagram shows additive skip connections (i.e. F(x) + x). One can also form concatenated skip connections, by "gluing together" F(x) and x into a single tensor. Give one advantage and one disadvantage of these, relative to additive connections.</summary>
#
# One advantage of concatenation: the subsequent layers can re-use middle representations; maintaining more information which can lead to better performance. Also, this still works if the tensors aren't exactly the same shape. One disadvantage: less compact, so there may be more weights to learn in subsequent layers.
#
# Crucially, both the addition and concatenation methods have the property of preserving information, to at least some degree of fidelity. For instance, you can [use calculus to show](https://theaisummer.com/skip-connections/#:~:text=residual%20skip%20connections.-,ResNet%3A%20skip%20connections%C2%A0via%C2%A0addition,-The%20core%20idea) that both methods will fix the vanishing gradients problem.
#
# </details>
#
# In this section, we'll do a more advanced version of the exercise in part 1. Rather than building a relatively simple network in which computation can be easily represented by a sequence of simple layers, we're going to build a more complex architecture which requires us to define nested blocks.
#
# We'll start by defining a few more `nn.Module` objects, which we hadn't needed before.
#

# %% [markdown]
# ## Sequential
#
# Firstly, now that we're working with large and complex architectures, we should create a version of `nn.Sequential`. As the name suggests, when an `nn.Sequential` is fed an input, it sequentially applies each of its submodules to the input, with the output from one module feeding into the next one.
#
# The implementation is given to you below. A few notes:
#
# - In initalization, we add to the `_modules` dictionary.
#   - This is a special type of dict called an **ordered dictionary**, which preserves the order of elements that get added (although Python sort-of does this now by default).
#   - When we call `self.parameters()`, this recursively goes through all modules in `self._modules`, and returns the params in those modules. This means we can nest sequentials within sequentials!
# - The special `__getitem__` and `__setitem__` methods determine behaviour when we get and set modules within the sequential.
# - The `repr` of the base class `nn.Module` already recursively prints out the submodules, so we don't need to write anything in `extra_repr`.
#   - To see how this works in practice, try defining a `Sequential` which takes a sequence of modules that you've defined above, and see what it looks like when you print it.
#
# Don't worry about deeply understanding this code. The main takeaway is that `nn.Sequential` is a useful list-like object to store modules, and apply them all sequentially.
#
# <details>
# <summary>Aside - initializing Sequential with an OrderedDict</summary>
#
# The actual `nn.Sequential` module can be initialized with an ordered dictionary, rather than a list of modules. For instance, rather than doing this:
#
# ```python
# seq = nn.Sequential(
#     nn.Linear(10, 20),
#     nn.ReLU(),
#     nn.Linear(20, 30)
# )
# ```
#
# we can do this:
#
# ```python
# seq = nn.Sequential(OrderedDict([
#     ("linear1", nn.Linear(10, 20)),
#     ("relu", nn.ReLU()),
#     ("linear2", nn.Linear(20, 30))
# ]))
# ```
#
# This is handy if we want to give each module an descriptive name.
#
# The `Sequential` implementation below doesn't allow the input to be an OrderedDict. As a bonus exercise, can you rewrite the `__init__`, `__getitem__` and `__setitem__` methods to allow the input to be an OrderedDict?
#
# </details>
#

# %%
from typing import Dict
from collections import OrderedDict


class Sequential(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.is_ordered_dict = len(modules) == 1 and isinstance(modules[0], OrderedDict)

        if self.is_ordered_dict:
            for key, mod in modules[0].items():
                self._modules[str(key)] = mod
        else:
            for index, mod in enumerate(modules):
                self._modules[str(index)] = mod

    def __getitem__(self, index: int | str) -> nn.Module:
        if self.is_ordered_dict and type(index) == str:
            return self._modules[index]
        elif self.is_ordered_dict and type(index) == int:
            return list(self._modules.values())[index]
        else:
            index %= len(self._modules)  # deal with negative indices
            return self._modules[str(index)]

    def __setitem__(self, index: int | str, module: nn.Module) -> None:
        if self.is_ordered_dict and type(index) == str:
            self._modules[index] = module
        elif self.is_ordered_dict and type(index) == int:
            key = list(self._modules.keys())[index]
            self._modules[key] = module
        else:
            index %= len(self._modules)  # deal with negative indices
            self._modules[str(index)] = module

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            x = mod(x)
        return x


# %% [markdown]
# ## BatchNorm2d
#
# Now, we'll implement our `BatchNorm2d`, the layer described in the documents you hopefully read above.
#
# Something which might have occurred to you as you read about batch norm - how does it work when in inference mode? It makes sense to normalize over a batch of multiple input data, but normalizing over a single datapoint doesn't make any sense! This is why we have to introduce a new PyTorch concept: **buffers**.
#
# Unlike `nn.Parameter`, a buffer is not its own type and does not wrap a `Tensor`. A buffer is just a regular `Tensor` on which you've called [self.register_buffer](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer) from inside a `nn.Module`. As an example, `self.register_buffer("variable_name", t.zeros(10))` will define an object `self.variable_name` and register it as a buffer.
#
# What is a buffer, and why is it different to a standard attribute or to a `nn.Parameter` object? The differences are as follows:
#
# - It is normally included in the output of `module.state_dict()`, meaning that `torch.save` and `torch.load` will serialize and deserialize it.
# - It is moved between devices when you call `model.to(device)`.
# - It is not included in `module.parameters`, so optimizers won't see or modify it. Instead, your module will modify it as appropriate within `forward`.
#
# Implementation note: when defining BatchNorm2d, register_buffer lines must be **in order** and **after** initializing self.weight and self.bias (as the `copy_weights` function relies on order matching exactly).
#

# %% [markdown]
# ### Train and Eval Modes
#
# This is your first implementation that needs to care about the value of `self.training`, which is set to True by default, and can be set to False by `self.eval()` or to True by `self.train()`.
#
# In training mode, you should use the mean and variance of the batch you're on, but you should also update a stored `running_mean` and `running_var` on each call to `forward` using the "momentum" argument as described in the PyTorch docs. Your `running_mean` shuld be intialized as all zeros; your `running_var` should be initialized as all ones. Also, you should keep track of `num_batches_tracked`.
#
# <details>
# <summary>Aside on <code>num_batches_tracked</code> (optional, unimportant)</summary>
#
# PyTorch uses this to calculate momentum for calculation of the moving averages in the event that the module is intialized with `momentum=None`, although you don't need to worry about this because you can assume that the momentum parameter will always be a float in our use cases; we're just keeping track of `num_batches_tracked` to be consistent with PyTorch's version of BatchNorm2d, and to make sure that our state dictionaries have the same items.
#
# </details>
#
# In eval mode, you should use the running mean and variance that you stored before (not the mean and variance from the current batch).
#

# %% [markdown]
# ### Exercise - implement `BatchNorm2d`
#
# ```yaml
# Difficulty: 🔴🔴🔴🔴⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to 20-40 minutes on this exercise.
#
# This is the most challenging module you'll have implemented so far. Getting all the dimensions and operations right can be tricky.
# ```
#
# Implement `BatchNorm2d` according to the [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html). Call your learnable parameters `weight` and `bias` for consistency with PyTorch.
#
# We're implementing it with `affine=True` and `track_running_stats=True`.
#
# A few tips (some of them are repeated from above):
#
# - Remember to differentiate between training mode, when `self.training=True` (you calculate mean & variance and update the running values) and eval mode, when `self.training=False` (you use the running values).
# - After you''ve normalized `x`, don't forget to multiply by `weight` and add `bias` (you might need to reshape these tensors so that they broadcast correctly).
#
# <details>
# <summary>Help - I don't understand which dimensions I should be taking the mean over.</summary>
#
# The input has shape `(batch, channels, height, width)` (where channels is synonymous with features). You want to calculate the mean and variance for each channel, meaning you should reduce over all dimensions except the first, e.g. `t.mean(x, dim=(0, 2, 3))`. You should then update the running mean and variance accordingly.
#
# Tip: use the argument `keepdim=True` in your mean and variance calculations, to make sure that the mean & variance still broadcast with the original input when you calculate `(x - mean) / std`.
#
# </details>
#
# <details>
# <summary>Help - I don't understand what the formula for updating the running mean / variance should be.</summary>
#
# You want `running_mean <- (1 - momentum) * running_mean + momentum * new_mean`. Again, make sure you get the dimensions right - all the tensors in this operation should be 1D, with shape `(num_features,)`.
#
# </details>
#
# If you're struggling with this implementation, we do recommend reading the solution, because there are lots of non-obvious ways this implementation can go wrong.
#

# %%
# einops.rearrange(t.arange(1.0, 4.0), "n -> 1 n 1 1")

# %%
# (x - mean) / (t.sqrt(var + 1e-05)), (x - mean) / (t.sqrt(var + 1e-05)) + einops.rearrange(t.arange(1.0, 4.0), "n -> n 1 1")

# %%
# x = t.randint(2, 8, (2, 3, 2, 2), dtype=t.float32)
# mean = einops.reduce(x, "b c h w -> 1 c 1 1", t.mean)
# var = einops.reduce(x, "b c h w -> 1 c 1 1", t.var)


# %%
class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))
        self.eps = eps
        self.momentum = momentum

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        # reshape x from batch,  channel, h, w to channel, batch, h, w
        # because we need to take mean by channels

        if self.training:
            # normalize by channel
            # we can also do t.mean(x, dim=(0, 2, 3))
            mean = einops.reduce(x, "b c h w -> 1 c 1 1", t.mean)
            # we can also do t.var(x, dim=(0, 2, 3))
            var = einops.reduce(x, "b c h w -> 1 c 1 1", t.var)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.squeeze()

            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var.squeeze()

            self.num_batches_tracked += 1
        else:
            mean = self.running_mean.reshape(1, self.num_features, 1, 1)
            var = self.running_var.reshape(1, self.num_features, 1, 1)

        normalized = (x - mean) / t.sqrt(var + self.eps)

        weight = einops.rearrange(self.weight, "c -> 1 c 1 1")
        bias = einops.rearrange(self.bias, "c -> 1 c 1 1")

        # multiply by adding dims to weight
        normalized = normalized * weight + bias

        # print(x.shape, normalized.shape, normalized.max(), normalized.min())

        return normalized

    def extra_repr(self) -> str:
        pass


# tests.test_batchnorm2d_module(BatchNorm2d)
# tests.test_batchnorm2d_forward(BatchNorm2d)
# tests.test_batchnorm2d_running_mean(BatchNorm2d)

# %% [markdown]
# ## AveragePool
#
# Let's end our collection of `nn.Module`s with an easy one 🙂
#
# The ResNet has a Linear layer with 1000 outputs at the end in order to produce classification logits for each of the 1000 classes. Any Linear needs to have a constant number of input features, but the ResNet is supposed to be compatible with arbitrary height and width, so we can't just do a pooling operation with a fixed kernel size and stride.
#
# Luckily, the simplest possible solution works decently: take the mean over the spatial dimensions. Intuitively, each position has an equal "vote" for what objects it can "see".
#

# %% [markdown]
# ### Exercise - implement `AveragePool`
#
# ```yaml
# Difficulty: 🔴⚪⚪⚪⚪
# Importance: 🔵🔵⚪⚪⚪
#
# You should spend up to 5-10 minutes on this exercise.
# ```
#
# This should be a pretty straightforward implementation; it doesn't have any weights or parameters of any kind, so you only need to implement the `forward` method.
#

# %%
# x = t.randint(1, 3, (1, 3, 2, 2), dtype=t.float32)
# b = BatchNorm2d(3)


# %%
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return einops.reduce(x, "b c h w -> b c", t.mean)


# %% [markdown]
# ## Building ResNet
#
# Now we have all the building blocks we need to start assembling your own ResNet! The following diagram describes the architecture of ResNet34 - the other versions are broadly similar. Unless otherwise noted, convolutions have a kernel_size of 3x3, a stride of 1, and a padding of 1. None of the convolutions have biases.
#
# <details>
# <summary>Question: there would be no advantage to enabling biases on the convolutional layers. Why?</summary>
#
# Every convolution layer in this network is followed by a batch normalization layer. The first operation in the batch normalization layer is to subtract the mean of each output channel. But a convolutional bias just adds some scalar `b` to each output channel, increasing the mean by `b`. This means that for any `b` added, the batch normalization will subtract `b` to exactly negate the bias term.
#
# </details>
#
# <details>
# <summary>Question: why is it necessary for the output of the left and right computational tracks in ResidualBlock to be the same shape?</summary>
#
# Because they're added together at the end of the tracks. If they weren't the same shape, then they couldn't be added together.
#
# </details>
#
# <details>
# <summary>Help - I'm confused about how the nested subgraphs work.</summary>
#
# The right-most block in the diagram, `ResidualBlock`, is nested inside `BlockGroup` multiple times. When you see `ResidualBlock` in `BlockGroup`, you should visualise a copy of `ResidualBlock` sitting in that position.
#
# Similarly, `BlockGroup` is nested multiple times (four to be precise) in the full `ResNet34` architecture.
#
# </details>
#
# <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/resnet-fixed.svg" width="900">
#

# %% [markdown]
# ### Exercise - implement `ResidualBlock`
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to 15-30 minutes on this exercise.
# ```
#
# Implement `ResidualBlock` by referring to the diagram. A few more notes on the left and right branches of this diagram:
#
# #### Left branch
#
# This branch has 2 convolutional layers. One of them applies a stride (the `first_stride` argument below), and changes the number of features from `in_feats -> out_feats`. The second one is a shape-preserving convolution, i.e. it has stride 1 and changes the number of features from `out_feats -> out_feats`.
#
# #### Right branch
#
# You can think of this branch as essentially a skip connection. But remember, we need to add this branch's output onto the left branch's output at the end of the residual block. If the left branch doesn't change the shape of its inputs (i.e. no strides, and `in_feats == out_feats`) then we can just use the identity operator for this right branch (you can use `nn.Identity` for this). But if either we have strides or a different number of output features, then we can't use the identity for the right branch. Instead we use what is essentially the closest approximation to the identity - a 1x1 convolution with stride & channel arguments chosen to match the shape of the left branch (followed by a batchnorm to standardize the output). Note, you may assume that if `first_stride == 1` then we always have number of input features equal to number of output features (meaning you can use the identity operator for this branch if and only if `first_stride == 1`).
#
# <details>
# <summary>Help - I'm completely stuck on parts of the architecture.</summary>
#
# In this case, you can use the following code to import your own `resnet34`, and inspect its architecture:
#
# ```python
# resnet = models.resnet34()
# print(torchinfo.summary(resnet, input_size=(1, 3, 64, 64)))
# ```
#
# This will generate output telling you the names of each module, as well as the parameter counts.
#
# Unfortunately, this function won't work on your own model if your model breaks when an image is passed through. Since a lot of the time mistakes in the architecture will mean your model doesn't work, you won't be able to use `torchinfo.summary` on your model. Instead, you should compare the models by printing them out.
#
# </details>
#

# %%
# Now we have all the building blocks we need to start assembling your own ResNet! The following diagram describes the architecture of ResNet34 - the other versions are broadly similar. Unless otherwise noted, convolutions have a kernel_size of 3x3, a stride of 1, and a padding of 1. None of the convolutions have biases.


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1, debug=False):
        """
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        """
        super().__init__()
        if debug:
            print(in_feats, out_feats, first_stride)

        self.left = Sequential(
            OrderedDict(
                [
                    # padding 1 and stride 3 = "same padding"  ((k - 1) / 2)
                    (
                        "conv1",
                        Conv2d(
                            in_feats,
                            out_feats,
                            kernel_size=3,
                            stride=first_stride,
                            padding=1,
                            debug=debug,
                        ),
                    ),
                    ("bn1", BatchNorm2d(out_feats)),
                    ("relu", ReLU()),
                    (
                        "conv2",
                        Conv2d(
                            out_feats, out_feats, kernel_size=3, stride=1, padding=1
                        ),
                    ),
                    ("bn2", BatchNorm2d(out_feats)),
                ]
            )
        )

        if first_stride == 1 and in_feats == out_feats:
            # this means we applied same padding above. so just do identity no need to change.
            self.right = Sequential(nn.Identity())
        # if we changed number of features, or had a stride != 1
        # do a convolution
        else:
            #  But if either we have strides or a different number of output features, then we can't use the identity for the right branch. Instead we use what is essentially the closest approximation to the identity - a 1x1 convolution with stride & channel arguments chosen to match the shape of the left branch (followed by a batchnorm to standardize the output). Note, you may assume that if `first_stride == 1` then we always have number of input features equal to number of output features (meaning you can use the identity operator for this branch if and only if `first_stride == 1`).
            self.right = Sequential(
                # note we keep padding 0 with kernel size 1. Identity ish??
                Conv2d(
                    in_feats,
                    out_feats,
                    kernel_size=1,
                    stride=first_stride,
                    padding=0,
                    debug=debug,
                ),
                BatchNorm2d(out_feats),
            )

        self.relu = ReLU()
        self.debug = debug

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        """
        if self.debug:
            print(x.shape)
        out = self.relu(self.left(x) + self.right(x))
        if self.debug:
            print(out.shape)
        return out


# tests.test_residual_block(ResidualBlock)

# %% [markdown]
# ### Exercise - implement `BlockGroup`
#
# ```yaml
# Difficulty: 🔴🔴🔴⚪⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to 10-15 minutes on this exercise.
# ```
#
# Implement `BlockGroup` according to the diagram.
#
# The number of channels changes from `in_feats` to `out_feats` in the first `ResidualBlock` (all subsequent blocks will have `out_feats` input channels and `out_feats` output channels).
#
# The `height` and `width` of the input will only be changed if `first_stride>1` (in which case it will be downsampled by exactly this amount).
#
# You can also read the docstring for a description of the input and output shapes.
#


# %%
class BlockGroup(nn.Module):
    def __init__(
        self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1, debug=False
    ):
        """An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride."""
        super().__init__()
        self.blocks = Sequential(
            ResidualBlock(in_feats, out_feats, first_stride=first_stride),
            *[ResidualBlock(out_feats, out_feats) for _ in range(n_blocks - 1)],
        )
        if debug:
            print(
                "[BlockGroup]",
                in_feats,
                "->",
                out_feats,
                ",",
                "first_stride =",
                first_stride,
            )

        self.debug = debug

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        out = self.blocks(x)
        if self.debug:
            print("[BlockGroup]", "Input", x.shape, "Output", out.shape)
        return out


# tests.test_block_group(BlockGroup)

# %% [markdown]
# ### Exercise - implement `ResNet34`
#
# ```yaml
# Difficulty: 🔴🔴🔴🔴⚪
# Importance: 🔵🔵🔵🔵⚪
#
# You should spend up to 25-30 minutes on this exercise.
#
# You may have to return to this and previous exercises, if you find a bug later.
# ```
#
# Last step! Assemble `ResNet34` using the diagram.
#
# <details>
# <summary>Help - I'm not sure how to construct each of the BlockGroups.</summary>
#
# Each BlockGroup takes arguments `n_blocks`, `in_feats`, `out_feats` and `first_stride`. In the initialisation of `ResNet34` below, we're given a list of `n_blocks`, `out_feats` and `first_stride` for each of the BlockGroups. To find `in_feats` for each block, it suffices to note two things:
#
# 1. The first `in_feats` should be 64, because the input is coming from the convolutional layer with 64 output channels.
# 2. The `out_feats` of each layer should be equal to the `in_feats` of the subsequent layer (because the BlockGroups are stacked one after the other; with no operations in between to change the shape).
#
# You can use these two facts to construct a list `in_features_per_group`, and then create your BlockGroups by zipping through all four lists.
#
# </details>
#
# <details>
# <summary>Help - I'm not sure how to construct the 7x7 conv at the very start.</summary>
#
# All the information about this convolution is given in the diagram, except for `in_channels`. Recall that the input to this layer is an RGB image. Can you deduce from this how many input channels your layer should have?
#
# </details>
#

# %%
# n_blocks_per_group=[3, 4, 6, 3]
# out_features_per_group=[64, 128, 256, 512]
# in_features_per_group = [3] + out_features_per_group[:-1]
# first_strides_per_group=[1, 2, 2, 2]
# n_classes=1000

# list(zip(
#     n_blocks_per_group,
#     in_features_per_group,
#     out_features_per_group,
#     first_strides_per_group,
# ))

# # %%
# class ResNet34(nn.Module):
#     def __init__(
#         self,
#         n_blocks_per_group=[3, 4, 6, 3],
#         out_features_per_group=[64, 128, 256, 512],
#         first_strides_per_group=[1, 2, 2, 2],
#         n_classes=1000,
#     ):
#         super().__init__()

#         self.n_blocks_per_group = n_blocks_per_group
#         self.out_features_per_group = out_features_per_group
#         self.first_strides_per_group = first_strides_per_group
#         self.n_classes = n_classes

#         self.blocks = Sequential(OrderedDict([
#             # Note that this is same padding. But stride = 2 reduces it to h / 2 and w / 2
#             # same padding = (k - 1) / 2 = 7 - 1 / 2 = 6 / 2 = 3
#             ("conv1", Conv2d(in_channels=3, out_channels=64, stride=2, kernel_size=7, padding=3)),
#             ("bn1", BatchNorm2d(num_features=64)),
#             ("relu", ReLU()),
#             # this is same padding. So stride = 2 will lead to height / 2 and width / 2
#             ("maxpool", MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             *[
#                 (f"layer{i+1}", BlockGroup(
#                     n_blocks=nb, in_feats=in_f, out_feats=out_f, first_stride=f_str
#                 ))
#                 for i, nb, in_f, out_f, f_str in zip(
#                     range(len(n_blocks_per_group)),
#                     n_blocks_per_group,
#                     [64] + out_features_per_group[:-1],
#                     out_features_per_group,
#                     first_strides_per_group,
#                 )
#             ],
#             ("avg", AveragePool()),
#             ("linear", Linear(in_features = out_features_per_group[-1], out_features=n_classes))
#         ])
#         )

#     def forward(self, x: t.Tensor) -> t.Tensor:
#         """
#         x: shape (batch, channels, height, width)
#         Return: shape (batch, n_classes)
#         """
#         return self.blocks(x)


# my_resnet = ResNet34()

# # %% [markdown]
# # Now that you've built your `ResNet34`, we'll copy weights over from PyTorch's pretrained resnet to yours. This is a good way to verify that you've designed the architecture correctly.
# #

# # %%
# def copy_weights(
#     my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet
# ) -> ResNet34:
#     """Copy over the weights of `pretrained_resnet` to your resnet."""

#     # Get the state dictionaries for each model, check they have the same number of parameters & buffers
#     mydict = my_resnet.state_dict()
#     pretraineddict = pretrained_resnet.state_dict()
#     # print(pretraineddict.keys(), mydict.keys())
#     assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

#     # Define a dictionary mapping the names of your parameters / buffers to their values in the pretrained model
#     state_dict_to_load = {
#         mykey: pretrainedvalue
#         for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(
#             mydict.items(), pretraineddict.items()
#         )
#     }

#     # Load in this dictionary to your model
#     my_resnet.load_state_dict(state_dict_to_load)

#     return my_resnet


# pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
# # my_resnet = copy_weights(my_resnet, pretrained_resnet)

# # %% [markdown]
# # This function uses the `state_dict()` method, which returns an `OrderedDict` (documentation [here](https://realpython.com/python-ordereddict/)) object containing all the parameter/buffer names and their values. State dicts can be extracted from models, saved to your filesystem (this is a common way to store the results of training a model), and can also be loaded back into a model using the `load_state_dict` method. (Note that you can also load weights using a regular Python `dict`, but since Python 3.7, the builtin `dict` is guaranteed to maintain items in the order they're inserted.)
# #
# # If the copying fails, this means that your model's layers don't match up with the layers in the PyTorch model implementation.
# #
# # To debug here, we've given you a helpful function `print_param_count` (from `utils.py`), which takes two models and prints out a stylized dataframe comparing the parameter names and shapes of each model. It will tell you when your model matches up with the PyTorch implementation. It can be used as follows:
# #

# # %%
# print_param_count(my_resnet, pretrained_resnet)

# # %%


# # %% [markdown]
# # You'll hopefully see something like the image below (the layer names not necessarily matching, but the parameter counts & shapes hopefully matching).
# #
# # <img src="https://raw.githubusercontent.com/callummcdougall/Fundamentals/main/images/resnet-compared.png" width="900">
# #
# # Tweaking your model until all the layers match up might be a difficult and frustrating exercise at times! However, it's a pretty good example of the kind of low-level model implementation and debugging that is important for your growth as ML engineers. We'll be doing a few more model-building exercises similar to these in later sections.
# #

# # %% [markdown]
# # ## Running Your Model
# #
# # We've provided you with some images for your model to classify:
# #

# # %%
# IMAGE_FILENAMES = [
#     "chimpanzee.jpg",
#     "golden_retriever.jpg",
#     "platypus.jpg",
#     "frogs.jpg",
#     "fireworks.jpg",
#     "astronaut.jpg",
#     "iguana.jpg",
#     "volcano.jpg",
#     "goofy.jpg",
#     "dragonfly.jpg",
# ]

# IMAGE_FOLDER = section_dir / "resnet_inputs"

# images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]

# # %% [markdown]
# # Our `images` are of type `PIL.Image.Image`, so we can just call them in a cell to display them.
# #

# # %%


# # %% [markdown]
# # We now need to define a `transform` object like we did for MNIST. We will use the same transforms to convert the PIL image to a tensor, and to normalize it. But we also want to resize the images to `height=224, width=224`, because not all of them start out with this size and we need them to be consistent before passing them through our model.
# #
# # In the normalization step, we'll use a mean of `[0.485, 0.456, 0.406]`, and a standard deviation of `[0.229, 0.224, 0.225]` (these are the mean and std dev of images from [ImageNet](https://www.image-net.org/)). Note that the means and std devs have three elements, because ImageNet contains RGB rather than monochrome images, and we're normalising over each of the three RGB channels separately.
# #

# # %%
# IMAGE_SIZE = 224
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]

# IMAGENET_TRANSFORM = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#         transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     ]
# )

# prepared_images = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0)

# assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)

# # %% [markdown]
# # ### Exercise - verify your model's predictions
# #
# # ```c
# # Difficulty: 🔴🔴⚪⚪⚪
# # Importance: 🔵🔵🔵⚪⚪
# #
# # You should spend up to ~10 minutes on this exercise.
# # ```
# #
# # Lastly, you should run your model with these prepared images, and verify that your predictions are the same as the model's predictions.
# #
# # You can do this by filling in the `predict` function below, then running the code. We've also provided you with a file `imagenet_labels.json` which you can use to get the actual classnames of imagenet data, and see what your model's predictions actually are.
# #

# # %%
# x = t.randint(2, 8, (2, 3))

# x, x.argmax()

# # %%
# def predict(model, images: t.Tensor) -> list[int]:
#     """
#     Returns the predicted class for each image (as a list of ints).
#     """
#     out = model(images)
#     classes = out.argmax(dim=1)
#     return classes


# with open(section_dir / "imagenet_labels.json") as f:
#     imagenet_labels = list(json.load(f).values())

# # Check your predictions match those of the pretrained model
# my_predictions = predict(my_resnet, prepared_images)
# pretrained_predictions = predict(pretrained_resnet, prepared_images)
# assert all(my_predictions == pretrained_predictions)
# print("All predictions match!")

# # Print out your predictions, next to the corresponding images
# for img, label in zip(images, my_predictions):
#     print(f"Class {label}: {imagenet_labels[label]}")
#     display(img)
#     print()

# # %% [markdown]
# # If you've done everything correctly, your version should give the same classifications, and the percentages should match at least to a couple decimal places.
# #
# # If it does, congratulations, you've now run an entire ResNet, using barely any code from `torch.nn`! The only things we used were `nn.Module` and `nn.Parameter`.
# #
# # If it doesn't, you get to practice model debugging! Remember to use the `utils.print_param_count` function that was provided.
# #
# # <details>
# # <summary>Help! My model is predicting roughly the same percentage for every category!</summary>
# #
# # This can indicate that your model weights are randomly initialized, meaning the weight loading process didn't actually take. Or, you reinitialized your model by accident after loading the weights.
# #
# # </details>
# #

# # %% [markdown]
# # ### Aside - hooks
# #
# # One problem you might have encountered is that your model outputs `NaN`s rather than actual numbers. When debugging this, it's useful to try and identify which module the error first appears in. This is a great use-case for **hooks**, which are something we'll be digging a lot more into during our mechanistic interpretability exercises later on.
# #
# # A hook is basically a function which you can attach to a particular `nn.Module`, which gets executed during your model's forward or backward passes. Here, we'll only consider forward hooks. A hook function's type signature is:
# #
# # ```python
# # def hook(module: nn.Module, inputs: list[t.Tensor], output: t.Tensor) -> None:
# #     pass
# # ```
# #
# # The `inputs` argument is a list of the inputs to the module (often just one tensor), and the `output` argument is the output of the module. This hook gets registered to a module by calling `module.register_forward_hook(hook)`. During forward passes, the hook function will run.
# #
# # Here is some code which will check for `NaN`s in the output of each module, and raise a `ValueError` if it finds any. We've also given you an example tiny network which produces a `NaN` in the output of the second layer, to demonstrate it on.
# #

# # %%
# class NanModule(nn.Module):
#     """
#     Define a module that always returns NaNs (we will use hooks to identify this error).
#     """

#     def forward(self, x):
#         return t.full_like(x, float("nan"))


# model = nn.Sequential(nn.Identity(), NanModule(), nn.Identity())


# def hook_check_for_nan_output(
#     module: nn.Module, input: tuple[t.Tensor], output: t.Tensor
# ) -> None:
#     """
#     Hook function which detects when the output of a layer is NaN.
#     """
#     if t.isnan(output).any():
#         raise ValueError(f"NaN output from {module}")


# def add_hook(module: nn.Module) -> None:
#     """
#     Register our hook function in a module.

#     Use model.apply(add_hook) to recursively apply the hook to model and all submodules.
#     """
#     module.register_forward_hook(hook_check_for_nan_output)


# def remove_hooks(module: nn.Module) -> None:
#     """
#     Remove all hooks from module.

#     Use module.apply(remove_hooks) to do this recursively.
#     """
#     module._backward_hooks.clear()
#     module._forward_hooks.clear()
#     module._forward_pre_hooks.clear()


# model = model.apply(add_hook)
# input = t.randn(3)

# try:
#     output = model(input)
# except ValueError as e:
#     print(e)

# model = model.apply(remove_hooks)

# # %% [markdown]
# # When you run this code, you should find it raising an error at the `NanModule`.
# #
# # > Important - when you're working with PyTorch hooks, make sure you remember to remove them at the end of each exercise! This is a classic source of bugs, and one of the things that make PyTorch hooks so janky. When we study TransformerLens in the next chapter, we'll use a version of hooks that is essentially the same under the hood, but comes with quite a few quality of life improvements!
# #

# # %% [markdown]
# # # 5️⃣ Bonus - Convolutions From Scratch
# #

# # %% [markdown]
# # > ### Learning objectives
# # >
# # > - Understand how array strides work, and why they're important for efficient linear operations
# # > - Learn how to use `as_strided` to perform simple linear operations like trace and matrix multiplication
# # > - Implement your own convolutions and maxpooling functions using stride-based methods
# #

# # %% [markdown]
# # This section is designed to get you familiar with the implementational details of layers like `Linear` and `Conv2d`. You'll be using libraries like `einops`, and functions like `torch.as_strided` to get a very low-level picture of how these operations work, which will help build up your overall understanding.
# #
# # Note that `torch.as_strided` isn't something which will come up explicitly in much of the rest of the course (unlike `einops`). The purpose of the stride exercises is more to give you an appreciation for what's going on under the hood, so that we can build layers of abstraction on top of that during the rest of this week (and by extension this course). I see this as analogous to how [many CS courses](https://cs50.harvard.edu/x/2023/) start by teaching you about languages like C and concepts like pointers and memory management before moving on to higher-level langauges like Python which abstract away these details. The hope is that when you get to the later sections of the course, you'll have the tools to understand them better.
# #

# # %%
# u = t.arange(10)
# v = t.arange(5)


# # repeat each element of u 5 times

# o1 = t.outer(u, v)
# o2 = u.as_strided(size=(10, 5), stride=(1, 0)) * v.as_strided((10, 5), stride=(0, 1))

# if t.all(t.isclose(o1, o2)):
#     print("Outer correct")
# else:
#     print("Outer incorrect")

# # matmul
# m = t.arange(10).view(5, 2)
# n = t.arange(20).view(2, 10)
# matmul1 = m @ n

# # m, m.as_strided(size=(5, 2, 10), stride=(1, 1, 0))
# matmul2 = m.as_strided(size=(5, 2, 10), stride=(2, 1, 0)) * n.as_strided(size=(5, 2, 10), stride=(0, 10, 1))
# matmul2 = einops.reduce(matmul2, "row_m row_m_col_m mul -> row_m mul", t.sum)

# if t.all(t.isclose(matmul1, matmul2)):
#     print("Matmul correct")
# else:
#     print("Matmul incorrect")

# # %%
# t.eye(4, 5)

# # %%
# rows = 5
# cols = 6
# m = t.ones(rows * cols, dtype=t.float32).reshape(1, 1, rows, cols)
# k_rows = 2
# k_cols = 5
# k = t.eye(k_rows, k_cols).reshape(1, 1, k_rows, k_cols)
# conv = conv2d(m, k)
# m, conv, k

# for r in range(rows):
#     for c in range(cols):
#         s = 0
#         for i in range(k_rows):
#             for j in range(k_cols):
#                 m_i = r + i
#                 m_j = c + j


# # %%


# # %%


# # %% [markdown]
# # ## Reading
# #
# # - [Python NumPy, 6.1 - `as_strided()`](https://www.youtube.com/watch?v=VlkzN00P0Bc) explains what array strides are.
# # - [`as_strided` and `sum` are all you need](https://jott.live/markdown/as_strided) gives an overview of how to use `as_strided` to perform array operations.
# # - [Advanced NumPy: Master stride tricks with 25 illustrated exercises](https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20) provides several clear and intuitive examples of `as_strided` being used to construct arrays.
# #

# # %% [markdown]
# # ## Basic stride exercises
# #
# # Array strides, and the `as_strided` method, are important to understand well because lots of linear operations are actually implementing something like `as_strided` under the hood.
# #
# # Run the following code, to define this tensor:
# #

# # %%
# test_input = t.tensor(
#     [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
#     dtype=t.float,
# )

# # %% [markdown]
# # This tensor is stored in a contiguous block in computer memory.
# #
# # We can call the `stride` method to get the strides of this particular array. Running `test_input.stride()`, we get `(5, 1)`. This means that we need to skip over one element in the storage of this tensor to get to the next element in the row, and 5 elements to get the next element in the column (because you have to jump over all 5 elements in the row). Another way of phrasing this: the `n`th element in the stride is the number of elements we need to skip over to move one index position in the `n`th dimension.
# #

# # %% [markdown]
# # ### Exercise - fill in the correct size and stride
# #
# # ```yaml
# # Difficulty: 🔴🔴🔴🔴⚪
# # Importance: 🔵🔵⚪⚪⚪
# #
# # You should spend up to ~30 minutes on these exercises collectively.
# #
# # as_strided exercises can be notoriously confusing and fiddly, so you should be willing to look at the solution if you're stuck. They are not the most important part of the material today.
# # ```
# #
# # In the exercises below, we will work with the `test_input` tensor above. You should fill in the `size` and `stride` arguments so that calling `test_input.as_strided` with these arguments produces the desired output. When you run the cell, the `for` loop at the end will iterate through the test cases and print out whether the test passed or failed.
# #
# # We've already filled in the first two as an example, along with illustrations explaining what's going on:
# #
# # <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/strides2.png" width="650">
# #
# # By the end of these examples, hopefully you'll have a clear idea of what's going on. If you're still confused by some of these, then the dropdown below the codeblock contains some annotations to explain the answers.
# #

# # %%


# # %%
# t.rand(4).shape

# # %%
# import torch as t
# from collections import namedtuple

# TestCase = namedtuple("TestCase", ["output", "size", "stride"])

# test_cases = [
#     TestCase(
#         output=t.tensor([0, 1, 2, 3]),
#         size=(4,),
#         stride=(1,),
#     ),
#     TestCase(
#         output=t.tensor([[0, 2], [5, 7]]),
#         size=(2, 2),
#         stride=(5, 2),
#     ),
#     TestCase(
#         output=t.tensor([0, 1, 2, 3, 4]),
#         size=(5,),
#         stride=(1,),
#     ),
#     TestCase(
#         output=t.tensor([0, 5, 10, 15]),
#         size=(4,),
#         stride=(5,),
#     ),
#     TestCase(
#         output=t.tensor([[0, 1, 2], [5, 6, 7]]),
#         size=(2, 3),
#         stride=(5, 1),
#     ),
#     TestCase(
#         output=t.tensor([[0, 1, 2], [10, 11, 12]]),
#         size=(2, 3),
#         stride=(10, 1),
#     ),
#     TestCase(
#         output=t.tensor([[0, 0, 0], [11, 11, 11]]),
#         size=(2, 3),
#         stride=(11, 0),
#     ),
#     TestCase(
#         output=t.tensor([0, 6, 12, 18]),
#         size=(4, ),
#         stride=(6, ),
#     ),
# ]

# for i, test_case in enumerate(test_cases):
#     if (test_case.size is None) or (test_case.stride is None):
#         print(f"Test {i} failed: attempt missing.")
#     else:
#         actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
#         if (test_case.output != actual).any():
#             print(f"Test {i} failed:")
#             print(f"Expected: {test_case.output}")
#             print(f"Actual: {actual}")
#         else:
#             print(f"Test {i} passed!")

# # %% [markdown]
# # ## Intermediate stride exercises
# #
# # Now that you're comfortable with the basics, we'll dive a little deeper with `as_strided`. In the last few exercises of this section, you'll start to implement some more challenging stride functions: trace, matrix-vector and matrix-matrix multiplication, just like we did for `einsum` in the previous section.
# #

# # %% [markdown]
# # ### Exercise - trace
# #
# # ```yaml
# # Difficulty: 🔴🔴⚪⚪⚪
# # Importance: 🔵🔵⚪⚪⚪
# #
# # You should spend up to 10-15 minutes on this exercise.
# #
# # Use the hint if you're stuck.
# # ```
# #

# # %%
# def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
#     """
#     Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
#     """
#     i, j = mat.shape
#     return mat.as_strided(size=(i, ), stride=(j + 1, )).sum()


# # tests.test_trace(as_strided_trace)

# # %% [markdown]
# # <details>
# # <summary>Hint</summary>
# #
# # The trace is the sum of all the elements you get from starting at `[0, 0]` and then continually stepping down and right one element. Use strides to create a 1D array which contains these elements.
# #
# # </details>
# #

# # %% [markdown]
# # ### Exercise - matrix-vector multiplication
# #
# # ```yaml
# # Difficulty: 🔴🔴🔴⚪⚪
# # Importance: 🔵🔵🔵⚪⚪
# #
# # You should spend up to 15-20 minutes on this exercise.
# #
# # The hints should be especially useful here if you're stuck. There are two hints available to you.
# # ```
# #
# # You should implement this using only `as_strided` and `sum` methods, and elementwise multiplication `*` - in other words, no matrix multiplication functions!
# #

# # %%
# m = t.randint(2, 5, (2, 10))
# v = t.arange(10)
# m @ v, m, v, (m * v.as_strided(size=(2, 10), stride=(0, 1))).sum(dim=1)

# # %%
# vec = t.tensor([0.0823, 1.2023, 0.0479, 2.0087])
# j,  = vec.shape

# vec.as_strided(size=(4, j), stride=(0, 1))

# # %%
# def as_strided_mv(
#     mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]
# ) -> Float[Tensor, "i"]:
#     """
#     Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
#     """
#     i, j = mat.shape
#     strideV = vec.stride()
#     v = vec.as_strided(size=(i, j), stride=(0, strideV[0]))
#     return (mat * v).sum(dim=1)


# # tests.test_mv(as_strided_mv)
# # tests.test_mv2(as_strided_mv)

# # %% [markdown]
# # <details>
# # <summary>Hint 1</summary>
# #
# # You want your output array to be as follows:
# #
# # $$
# # \text{output}[i] = \sum_j \text{mat}[i, j] \times \text{vector}[j]
# # $$
# #
# # so first try to create an array with:
# #
# # $$
# # \text{arr}[i, j] = \text{mat}[i, j] \times \text{vector}[j]
# # $$
# #
# # then you can calculate `output` by summing over the second dimension of `arr`.
# #
# # </details>
# #
# # <details>
# # <summary>Hint 2</summary>
# #
# # First try to use strides to create `vec_expanded` such that:
# #
# # $$
# # \text{vec\_expanded}[i, j] = \text{vec}[j]
# # $$
# #
# # We can then compute:
# #
# # $$
# # \begin{align}
# # \text{arr}[i, j] &= \text{mat}[i, j] \times \text{vec\_expanded}[i, j] \\
# # \text{output}[i, j] &= \sum_j \text{arr}[i, j]
# # \end{align}
# # $$
# #
# # with the first equation being a simple elementwise multiplication, and the second equation being a sum over the second dimension.
# #
# # </details>
# #
# # <details>
# # <summary>Help - I'm passing the first test, but failing the second.</summary>
# #
# # It's possible that the input matrices you recieve could themselves be the output of an `as_strided` operation, so that they're represented in memory in a non-contiguous way. Make sure that your `as_strided `operation is using the strides from the original input arrays, i.e. it's not just assuming the last element in the `stride()` tuple is 1.
# #
# # </details>
# #

# # %% [markdown]
# # ### Exercise - matrix-matrix multiplication
# #
# # ```yaml
# # Difficulty: 🔴🔴🔴🔴⚪
# # Importance: 🔵🔵🔵⚪⚪
# #
# # You should spend up to 15-20 minutes on this exercise.
# #
# # The hints should be especially useful here if you're stuck. There are two hints available to you.
# # ```
# #
# # Like the previous function, this should only involve `as_strided`, `sum`, and pointwise multiplication.
# #

# # %%
# matA = t.randint(3, (2, 3))
# matB = t.randint(2, (3, 4))

# i, j = matA.shape
# j, k = matB.shape

# # final shape has to be i x k

# strideA = matA.stride()
# strideB = matB.stride()

# stA = matA.as_strided(size=(i, k, j), stride=(j, 0, 1))
# stB = matB.as_strided(size=(i, k, j), stride=(0, j, 1))
# matA @ matB, (stA * stB).sum(dim=-1)

# # %%


# # %%


# # %%
# def as_strided_mm(
#     matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]
# ) -> Float[Tensor, "i k"]:
#     """
#     Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
#     """
#     # convert both to a i x k x j matrix
#     i, j = matA.shape
#     j, k = matB.shape

#     # final shape has to be i x k
#     strideA = matA.stride()
#     strideB = matB.stride()

#     stA = matA.as_strided(size=(i, k, j), stride=(strideA[0], 0, strideA[1]))
#     stB = matB.as_strided(size=(i, k, j), stride=(0, strideB[1], strideB[0]))
#     return (stA * stB).sum(dim=-1)


# # tests.test_mm(as_strided_mm)
# # tests.test_mm2(as_strided_mm)

# # %%


# # %% [markdown]
# # <details>
# # <summary>Hint 1</summary>
# #
# # If you did the first one, this isn't too dissimilar. We have:
# #
# # $$
# # \text{output}[i, k] = \sum_j \text{matA}[i, j] \times \text{matB}[j, k]
# # $$
# #
# # so in this case, try to create an array with:
# #
# # $$
# # \text{arr}[i, j, k] = \text{matA}[i, j] \times \text{matB}[j, k]
# # $$
# #
# # then sum this array over `j` to get our output.
# #
# # We need to create expanded versions of both `matA` and `matB` in order to take this product.
# #
# # </details>
# #
# # <details>
# # <summary>Hint 2</summary>
# #
# # We want to compute
# #
# # $$
# # \text{matA\_expanded}[i, j, k] = \text{matA}[i, j]
# # $$
# #
# # so our stride for `matA` should be `(matA.stride(0), matA.stride(1), 0)`.
# #
# # A similar idea applies for `matB`.
# #
# # </details>
# #

# # %% [markdown]
# # ## conv1d minimal
# #
# # Here, we will implement the PyTorch `conv1d` function, which can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html). We will start with a simple implementation where `stride=1` and `padding=0`, with the other arguments set to their default values.
# #
# # Firstly, some explanation of `conv1d` in PyTorch. The `1` in `1d` here refers to the number of dimensions along which we slide the weights (also called the kernel) when we convolve. Importantly, it does not refer to the number of dimensions of the tensors that are being used in our calculations. Typically the input and kernel are both 3D:
# #
# # - `input.shape = (batch, in_channels, width)`
# # - `kernel.shape = (out_channels, in_channels, kernel_width)`
# #
# # A typical convolution operation is illustrated in the sketch below. Some notes on this sketch:
# #
# # - The `kernel_width` dimension of the kernel slides along the `width` dimension of the input. The `output_width` of the output is determined by the number of kernels that can be fit inside it; the formula can be seen in the right part of the sketch.
# # - For each possible position of the kernel inside the model (i.e. each freezeframe position in the sketch), the operation happening is as follows:
# #   - We take the product of the kernel values with the corresponding input values, and then take the sum
# #   - This gives us a single value for each output channel
# #   - These values are then passed into the output tensor
# # - The sketch assumes a batch size of 1. To generalise to a larger batch number, we can just imagine this operation being repeated identically on every input.
# #
# # <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ch0-conv1d-general.png" width=1050>
# #

# # %% [markdown]
# # ### A note on `out_channels`
# #
# # The out_channels in a conv2d layer denotes the number of filters the layer uses. Each filter detects specific features in the input, producing an output with as many channels as filters.
# #
# # This number isn't tied to the input image's channels but is a design choice in the neural network architecture. Commonly, powers of 2 are chosen for computational efficiency, and deeper layers might have more channels to capture complex features. Additionally, this parameter is sometimes chosen based on the heuristic of wanting to balance the parameter count / compute for each layer - which is why you often see `out_channels` growing as the size of each feature map gets smaller.
# #

# # %% [markdown]
# # ### Exercise - implement minimal 1D conv (part 1)
# #
# # ```yaml
# # Difficulty: 🔴🔴🔴🔴⚪
# # Importance: 🔵🔵⚪⚪⚪
# #
# # You should spend up to 15-20 minutes on this exercise.
# #
# # Use the diagram in the dropdown below, if you're stuck.
# # ```
# #
# # Below, you should implement `conv1d_minimal`. This is a function which works just like `conv1d`, but takes the default stride and padding values (these will be added back in later). You are allowed to use `as_strided` and `einsum`.
# #
# # Because this is a difficult exercise, we've given you a "simplified" function to implement first. This gets rid of the batch dimension, and input & output channel dimensions, so you only have to think about `x` and `weights` being one-dimensional tensors:
# #
# # <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ch0-conv1d-A.png" width=620>
# #

# # %%
# from torch import conv1d


# x = t.arange(10)
# k = t.arange(4)
# w, = x.shape
# kw, = k.shape
# output_width = w - kw + 1
# stride_input = x.stride()
# stride_k = k.stride()

# view_x = x.as_strided(size=(output_width, kw), stride=(stride_input[0], 1))
# view_k = k.as_strided(size=(output_width, kw), stride=(0, 1))

# # einops.einsum(view_x, view_k, "outw kw, outw kw -> outw"), t.conv1d(x.reshape(1, 1, w), k.reshape(1, 1, kw))
# x, view_x

# # %%
# def conv1d_minimal_simple(
#     x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]
# ) -> Float[Tensor, "ow"]:
#     """
#     Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

#     Simplifications: batch = input channels = output channels = 1.

#     x: shape (width,)
#     weights: shape (kernel_width,)

#     Returns: shape (output_width,)
#     """

#     w, = x.shape
#     kw, = weights.shape
#     output_width = w - kw + 1
#     stride_input = x.stride()

#     view_x = x.as_strided(size=(output_width, kw), stride=(stride_input[0], 1))
#     view_k = weights.as_strided(size=(output_width, kw), stride=(0, 1))

#     return einops.einsum(view_x, view_k, "outw kw, outw kw -> outw")


# # tests.test_conv1d_minimal_simple(conv1d_minimal_simple)

# # %% [markdown]
# # <details>
# # <summary>If you're stuck on <code>conv1d_minimal_simple</code>, click here to see a diagram which should help.</summary>
# #
# # This diagram illustrates the striding operation you'll need to perform on `x`. Once you do this, it's just a matter of using the right `einsum` operation to get the output.
# #
# # <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ch0-conv1d-explained.png" width=800>
# # </details>
# #

# # %% [markdown]
# # ### Exercise - implement minimal 1D conv (part 2)
# #
# # ```yaml
# # Difficulty: 🔴🔴🔴🔴⚪
# # Importance: 🔵🔵⚪⚪⚪
# #
# # You should spend up to 15-25 minutes on this exercise.
# # ```
# #
# # Once you've implemented this function, you should now adapt it to make a "full version", which includes batch, in_channel and out_channel dimensions. If you're stuck, the dropdowns provide hints for how each of these new dimensions should be handled.
# #

# # %%
# b = 1
# ic = 2
# w = 3
# oc = 3
# kw = 2

# x = t.randint(10, (b, ic, w))
# weights = t.randint(3, (oc, ic, kw))
# output_width = w - kw + 1 # 3 - 2 + 1 = 2

# stride_x = x.stride()
# stride_weights = weights.stride()

# # we want each of the oc kernels to see the ic, w strided matrix
# # also each kernel will result in output_width number of items
# # so we will have batch, each item in the batch sees oc kernels
# # each kernel causes output_width outputs
# x_strided = x.as_strided(size=(b, oc, output_width, ic, kw), stride=(stride_x[0], 0,  stride_x[2], stride_x[1], stride_x[2]))
# weights_strided = weights.as_strided(size=(b, oc, output_width, ic, kw), stride=(0, stride_weights[0], 0, stride_weights[1], stride_weights[2]))

# print(f"Input channels: {ic}")
# print(f"Output channels: {oc}")
# print(f"Each kernel has this many steps for each of the ic input channels): {output_width}")

# # weights, weights_strided
# # (x_strided * weights_strided).sum(dim=(-1, -2)), conv1d(x, weights)
# # x_strided = x.as_strided(size=(b, ic, output_width, kw), stride=(stride_x[0], stride_x[1], stride_x[2], stride_x[2]))
# x, x_strided

# # x is b, ic, output_width, kw
# # weights is oc ic kw
# # x, x_strided, weights, einops.einsum(x_strided, weights, "b ic ow kw, oc ic kw -> b oc ow")
# # x, x_strided, weights, weights.unsqueeze(0).unsqueeze(3).expand(x_strided.shape[0], weights.shape[0], weights.shape[1], x_strided.shape[2], weights.shape[2])


# # %%
# def conv1d_minimal(
#     x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]
# ) -> Float[Tensor, "b oc ow"]:
#     """
#     Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

#     x: shape (batch, in_channels, width)
#     weights: shape (out_channels, in_channels, kernel_width)

#     Returns: shape (batch, out_channels, output_width)
#     """
#     b, ic, w = x.shape
#     oc, ic, kw, = weights.shape
#     output_width = w - kw + 1

#     stride_x = x.stride()
#     stride_weights = weights.stride()

#     # we want each of the oc kernels to see the ic, w strided matrix
#     # also each kernel will result in output_width number of items
#     # so we will have batch, each item in the batch sees oc kernels
#     # each kernel causes output_width outputs
#     # x_strided = x.as_strided(size=(b, oc, output_width, ic, kw), stride=(stride_x[0], 0,  stride_x[2], stride_x[1], stride_x[2]))
#     # weights_strided = weights.as_strided(size=(b, oc, output_width, ic, kw), stride=(0, stride_weights[0], 0, stride_weights[1], stride_weights[2]))

#     # print(f"Input channels: {ic}")
#     # print(f"Output channels: {oc}")
#     # print(f"Each kernel has this many steps for each of the ic input channels): {output_width}")

#     # x, x_strided
#     # weights, weights_strided
#     # return (x_strided * weights_strided).sum(dim=(-1, -2))

#     # from solution:
#     x_strided = x.as_strided(size=(b, ic, output_width, kw), stride=(stride_x[0], stride_x[1], stride_x[2], stride_x[2]))

#     return einops.einsum(x_strided, weights, "b ic ow kw, oc ic kw -> b oc ow")


# # tests.test_conv1d_minimal(conv1d_minimal)

# # %% [markdown]
# # <details>
# # <summary>Help - I'm stuck on going from <code>conv1d_minimal_simple</code> to <code>conv1d_minimal</code>.</summary>
# #
# # The principle is the same as before. In your function, you should:
# #
# # - Create a strided version of `x` by adding a dimension of length `output_width` and with the same stride as the `width` stride of `x` (the purpose of which is to be able to do all the convolutions at once).
# # - Perform an einsum between this strided version of `x` and `weights`, summing over the appropriate dimensions.
# #
# # The way each of the new dimensions `batch`, `out_channels` and `in_channels` are handled is as follows:
# #
# # - `batch` - this is an extra dimension for `x`, it is _not_ summed over when creating `output`.
# # - `out_channels` - this is an extra dimension for `weights`, it is _not_ summed over when creating `output`.
# # - `in_channels` - this is an extra dimension for `weights` _and_ for `x`, it _is_ summed over when creating `output`.
# # </details>
# #

# # %% [markdown]
# # ## conv2d minimal
# #
# # 2D convolutions are conceptually similar to 1D. The only difference is in how you move the kernel across the tensor as you take your convolution. In this case, you will be moving the tensor across two dimensions:
# #
# # <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ch0-conv2d-B.png" width=850>
# #
# # For this reason, 1D convolutions tend to be used for signals (e.g. audio), 2D convolutions are used for images, and 3D convolutions are used for 3D scans (e.g. in medical applications).
# #

# # %% [markdown]
# # ### Exercise - implement 2D minimal convolutions
# #
# # ```yaml
# # Difficulty: 🔴🔴🔴🔴⚪
# # Importance: 🔵🔵⚪⚪⚪
# #
# # You should spend up to 20-25 minutes on this exercise.
# #
# # Use the diagram in the dropdown below, if you're stuck.
# # ```
# #
# # You should implement `conv2d` in a similar way to `conv1d`. Again, this is expected to be difficult and there are several hints you can go through.
# #

# # %%
# from torch import conv2d

# b = 1
# ic = 2
# oc = 3
# h = 3
# w = 4
# kh = 2
# kw = 2

# oh = h - kh + 1
# ow = w - kw + 1

# x = t.randint(10, (b, ic, h, w))
# k = t.randint(3, (oc, ic, kh, kw))

# sx_b, sx_ic, sx_kh, sx_kw = x.stride()

# sk_oc, sk_ic, sk_h, sk_w = k.stride()
# # each kernel's output channel
# # will see each input channel
# # we will first create both to be the same sized tensors
# # print()
# x_strided = x.as_strided(size=(b, oc, ic, oh, ow, kh, kw), stride=(sx_b, 0, sx_ic, sx_h, sx_w, sx_h, sx_w))

# einops.einsum(x_strided, k, "b oc ic oh ow kh kw, oc ic kh kw -> b oc oh ow"), conv2d(x, k)

# # %%


# # %%


# # %%


# # %%
# def conv2d_minimal(
#     x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]
# ) -> Float[Tensor, "b oc oh ow"]:
#     """
#     Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

#     x: shape (batch, in_channels, height, width)
#     weights: shape (out_channels, in_channels, kernel_height, kernel_width)

#     Returns: shape (batch, out_channels, output_height, output_width)
#     """
#     b, ic, h, w = x.shape
#     oc, ic, kh, kw = weights.shape

#     oh = h - kh + 1
#     ow = w - kw + 1

#     sx_b, sx_ic, sx_h, sx_w = x.stride()
#     # each kernel's output channel
#     # will see each input channel
#     # we will first create both to be the same sized tensors
#     # print()
#     x_strided = x.as_strided(size=(b, ic, oh, ow, kh, kw), stride=(sx_b, sx_ic, sx_h, sx_w, sx_h, sx_w))

#     return einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")


# # tests.test_conv2d_minimal(conv2d_minimal)

# # %% [markdown]
# # <details>
# # <summary>Hint & diagram</summary>
# #
# # You should be doing the same thing that you did for the 1D version. The only difference is that you're introducing 2 new dimensions to your strided version of x, rather than 1 (their sizes should be `output_height` and `output_width`, and their strides should be the same as the original `height` and `width` strides of `x` respectively).
# #
# # <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ch0-conv2d-explained.png" width=1000>
# # </details>
# #

# # %% [markdown]
# # ## Padding
# #

# # %% [markdown]
# # For a full version of `conv`, and for `maxpool` (which will follow shortly), you'll need to implement `pad` helper functions. PyTorch has some very generic padding functions, but to keep things simple and build up gradually, we'll write 1D and 2D functions individually.
# #

# # %% [markdown]
# # ### Exercise - implement padding
# #
# # ```yaml
# # Difficulty: 🔴🔴⚪⚪⚪
# # Importance: 🔵🔵⚪⚪⚪
# #
# # You should spend up to 15-20 minutes on this exercise, and the next.
# # ```
# #
# # Tips:
# #
# # - Use the `new_full` method of the input tensor. This is a clean way to ensure that the output tensor is on the same device as the input, and has the same dtype.
# # - You can use three dots to denote slicing over multiple dimensions. For instance, `x[..., 0]` will take the `0th` slice of `x` along its last dimension. This is equivalent to `x[:, 0]` for 2D, `x[:, :, 0]` for 3D, etc.
# #

# # %%
# x = [1, 2, 3]

# x[0:None]

# # %%
# def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
#     """Return a new tensor with padding applied to the edges.

#     x: shape (batch, in_channels, width), dtype float32

#     Return: shape (batch, in_channels, left + right + width)
#     """
#     b, c, w = x.shape
#     n = x.new_full(size=(b, c, left + right + w), fill_value=pad_value)
#     # now fill all except left, right
#     end_idx = None if right == 0 else -right
#     start_idx = left
#     n[..., start_idx:end_idx] = x[..., :]

#     return n


# # tests.test_pad1d(pad1d)
# # tests.test_pad1d_multi_channel(pad1d)

# # %% [markdown]
# # <details>
# # <summary>Help - I get <code>RuntimeError: The expanded size of the tensor (0) must match ...</code></summary>
# #
# # This might be because you've indexed with `left : -right`. Think about what will happen here when `right` is zero.
# #
# # </details>
# #

# # %% [markdown]
# # Once you've passed the tests, you can implement the 2D version:
# #

# # %%
# x = t.randint(10, (3, 4))

# x, x[0:2, 1:4]

# # %%
# def pad2d(
#     x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float
# ) -> t.Tensor:
#     """Return a new tensor with padding applied to the edges.

#     x: shape (batch, in_channels, height, width), dtype float32

#     Return: shape (batch, in_channels, top + height + bottom, left + width + right)
#     """
#     b, c, h, w = x.shape
#     n = x.new_full(size=(b, c, top + bottom + h, left + right + w), fill_value=pad_value)

#     start_left_idx = left
#     end_right_idx = None if right == 0 else -right
#     start_top_idx = top
#     end_top_idx = None if bottom == 0 else -bottom

#     n[..., start_top_idx:end_top_idx, start_left_idx:end_right_idx] = x
#     return n


# # tests.test_pad2d(pad2d)
# # tests.test_pad2d_multi_channel(pad2d)

# # %% [markdown]
# # ## Full convolutions
# #
# # Now, you'll extend `conv1d` to handle the `stride` and `padding` arguments.
# #
# # `stride` is the number of input positions that the kernel slides at each step. `padding` is the number of zeros concatenated to each side of the input before the convolution.
# #
# # Output shape should be `(batch, output_channels, output_length)`, where output_length can be calculated as follows:
# #
# # $$
# # \text{output\_length} = \left\lfloor\frac{\text{input\_length} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} \right\rfloor + 1
# # $$
# #
# # Verify for yourself that the forumla above simplifies to the formula we used earlier when padding is 0 and stride is 1.
# #
# # Docs for pytorch's `conv1d` can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html).
# #

# # %% [markdown]
# # ### Exercise - implement 1D convolutions
# #
# # ```yaml
# # Difficulty: 🔴🔴🔴🔴⚪
# # Importance: 🔵🔵⚪⚪⚪
# #
# # You should spend up to 20-25 minutes on this exercise.
# # ```
# #

# # %%
# from math import floor
# from torch import conv1d as cv


# def conv1d(
#     x: Float[Tensor, "b ic w"],
#     weights: Float[Tensor, "oc ic kw"],
#     stride: int = 1,
#     padding: int = 0,
# ) -> Float[Tensor, "b oc ow"]:
#     """
#     Like torch's conv1d using bias=False.

#     x: shape (batch, in_channels, width)
#     weights: shape (out_channels, in_channels, kernel_width)

#     Returns: shape (batch, out_channels, output_width)
#     """
#     b, ic, w = x.shape
#     oc, ic, kw = weights.shape
#     ow = ((w + 2 * padding - kw) // stride) + 1

#     # first pad
#     padded = pad1d(x, padding, padding, 0.0)

#     # print("Input")
#     # print(x)
#     # print("Stride")
#     # print(stride)
#     # print("Kernel")
#     # print(k)
#     # print("Padded input")
#     # print(padded)

#     s_b, s_ic, s_w = padded.stride()
#     padded = padded.as_strided(size=(b, ic, ow, kw), stride=(s_b, s_ic, stride, s_w))

#     # print("Strided padded input")
#     # print(padded)
#     # print(cv(x, weights, padding=padding, stride=stride))
#     return einops.einsum(padded, weights, "b ic ow kw, oc ic kw -> b oc ow")


# # x = t.randint(10, (1, 2, 3))
# # k = t.randint(3, (2, 2, 2))
# # padding = 2

# # conv1d(x, k, padding=padding, stride=2)

# # tests.test_conv1d(conv1d)

# # %% [markdown]
# # <details>
# # <summary>Hint - dealing with padding</summary>
# #
# # As the first line of your function, replace `x` with the padded version of `x`. This way, you won't have to worry about accounting for padding in the rest of the function (e.g. in the formula for the output width).
# #
# # </details>
# #
# # <details>
# # <summary>Hint - dealing with strides</summary>
# #
# # The following diagram shows how you should create the strided version of `x` differently, if you have a stride of 2 rather than the default stride of 1.
# #
# # <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ch0-conv1d-strides.png" width="850">
# #
# # Remember, you'll need a new formula for `output_width` (see formula in the [documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) for help with this, or see if you can derive it without help).
# #
# # </details>
# #

# # %% [markdown]
# # ### Exercise - implement 2D convolutions
# #
# # ```yaml
# # Difficulty: 🔴🔴🔴🔴⚪
# # Importance: 🔵🔵⚪⚪⚪
# #
# # You should spend up to 20-25 minutes on this exercise.
# # ```
# #
# # A recurring pattern in these 2d functions is allowing the user to specify either an int or a pair of ints for an argument: examples are stride and padding. We've provided some type aliases and a helper function to simplify working with these.
# #

# # %%
# IntOrPair = int | tuple[int, int]
# Pair = tuple[int, int]


# def force_pair(v: IntOrPair) -> Pair:
#     """Convert v to a pair of int, if it isn't already."""
#     if isinstance(v, tuple):
#         if len(v) != 2:
#             raise ValueError(v)
#         return (int(v[0]), int(v[1]))
#     elif isinstance(v, int):
#         return (v, v)
#     raise ValueError(v)


# # Examples of how this function can be used:

# for v in [(1, 2), 2, (1, 2, 3)]:
#     try:
#         print(f"{v!r:9} -> {force_pair(v)!r}")
#     except ValueError:
#         print(f"{v!r:9} -> ValueError")

# # %% [markdown]
# # Finally, you can implement a full version of `conv2d`. If you've done the full version of `conv1d`, and you've done `conv2d_minimal`, then you should be able to pull code from here to help you.
# #

# # %%
# x = t.randint(10, (3, 3))
# x, t.nn.functional.pad(x, pad=(1, 1, 3, 2))

# # %%
# from torch import conv2d as cv

# def conv2d(
#     x: Float[Tensor, "b ic h w"],
#     weights: Float[Tensor, "oc ic kh kw"],
#     stride: IntOrPair = 1,
#     padding: IntOrPair = 0,
# ) -> Float[Tensor, "b oc oh ow"]:
#     """
#     Like torch's conv2d using bias=False

#     x: shape (batch, in_channels, height, width)
#     weights: shape (out_channels, in_channelsss, kernel_height, kernel_width)

#     Returns: shape (batch, out_channels, output_height, output_width)
#     """
#     b, ic, h, w = x.shape
#     oc, ic, kh, kw = weights.shape


#     # padding and stride are also dimension wise. not normal x and y.

#     sh, sw = force_pair(stride)
#     ph, pw = force_pair(padding)


#     padded = pad2d(x, pw, pw, ph, ph, 0.0)

#     sx_b, sx_ic, sx_h, sx_w = padded.stride()

#     ow = ((w + 2 * pw - kw) // sw) + 1
#     oh = ((h + 2 * ph - kh) // sh) + 1


#     padded_strided = padded.as_strided(size=(b, ic, oh, ow, kh, kw), stride=(sx_b, sx_ic, sh * sx_h, sw, sx_h, sx_w))

#     return einops.einsum(padded_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")


# # tests.test_conv2d(conv2d)
# # x =  t.randint(10, size=(1, 2, 3, 4))
# # k = t.randint(3, size=(1, 2, 2, 2))

# # padding = (2, 3)
# # stride = (1, 3)
# # x, conv2d(x = x, weights=k, padding=padding, stride=stride), cv(input=x, weight=k, padding=padding, stride=stride)

# # %% [markdown]
# # ## Max pooling
# #
# # We have just one function left now - **max pooling**. You can review the [TowardsDataScience](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) post from earlier to understand max pooling better.
# #
# # A "max pooling" layer is similar to a convolution in that you have a window sliding over some number of dimensions. The main difference is that there's no kernel: instead of multiplying by the kernel and adding, you just take the maximum.
# #
# # The way multiple channels work is also different. A convolution has some number of input and output channels, and each output channel is a function of all the input channels. There can be any number of output channels. In a pooling layer, the maximum operation is applied independently for each input channel, meaning the number of output channels is necessarily equal to the number of input channels.
# #

# # %% [markdown]
# # ### Exercise - implement 2D max pooling
# #
# # ```yaml
# # Difficulty: 🔴🔴⚪⚪⚪
# # Importance: 🔵🔵🔵⚪⚪
# #
# # You should spend up to 10-15 minutes on this exercise.
# # ```
# #
# # Implement `maxpool2d` using `torch.as_strided` and `torch.amax` (= max over axes) together. Your version should behave the same as the PyTorch version, but only the indicated arguments need to be supported.
# #

# # %%
# def maxpool2d(
#     x: Float[Tensor, "b ic h w"],
#     kernel_size: IntOrPair,
#     stride: IntOrPair | None = None,
#     padding: IntOrPair = 0,
# ) -> Float[Tensor, "b ic oh ow"]:
#     """
#     Like PyTorch's maxpool2d.

#     x: shape (batch, channels, height, width)
#     stride: if None, should be equal to the kernel size

#     Return: (batch, channels, output_height, output_width)
#     """
#     pass


# # tests.test_maxpool2d(maxpool2d)
