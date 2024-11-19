import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # image is [B, C, H, W]
        # embed_dims = number of kernels (aka out_channels)
        # we convert it to [B, embed_dim, patch_size, patch_size]
        # we use a convolution to generate the desired embedding size from a passed image
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor):
        _, _, height, width = pixel_values.shape
        # do the convolution and add position embeddings to it
        # convolution converts from b, c, h, w -> b, embed_dim, patch_size, patch_size
        # num_patches = (height // patch_size) ** 2
        patch_embeds: torch.Tensor = self.patch_embedding(pixel_values)

        # we flatten it to b, embed_dim, num_patches to be able to add it to pos embeddings later
        embeddings = patch_embeds.flatten(2)

        # position_embedding is num_positions x embed_dims
        # we will convert above patch_Embed from b, embed_dim, num_patches to
        # b, num_patches, embed_dim
        embeddings = embeddings.transpose(1, 2)

        # now add positional embeddings
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SiglipAttention(nn.Module):
    """Multi head attn from attn is all you need paper"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, norm1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # norm1 is b, n_patches, embed_dim
        # split into
        # num_patches can be thought of as seq_len
        batch_size, seq_len, _ = norm1.size()

        # b, n_patches, embed_dim
        key_states = self.k_proj(norm1)
        # b, n_patches, embed_dim
        query_states = self.q_proj(norm1)
        # b, n_patches, embed_dim
        value_states = self.v_proj(norm1)

        # convert to [batch_size, self.num_heads, seq_len, self.head_dim]
        # so now each head will see b examples
        # each with [head_dim, n_patches, num_heads]
        # the reason we don't directly do a .view(batch_size, num_heads, seq_len, head_dim)
        # is because view will just take the tensor laid out contiguously in memory
        # and split it first by head dim, then by seq_len
        # this will cause overlap between tokens
        # we only want to first:
        # split the individual tokens into diff heads to get an element of a batch with [seq_len, num_heads, head_dim] dimensions
        # right now, we have a seq_len where each token is split into num_heads x head_dim matrix
        # but we want to create num_heads sequences with seq_len x head_dim matrix
        # NOTE: play around in the terminal. Note that printing a tensor will cause a shit output.
        # bring the outputs into vscode and reformat such that all "rows" are laid out horizontally
        # Try this code and print outputs
        """
        # Initialize a tensor with shape (batch_size, seq_len, embed_dim)
        batch_size = 2
        seq_len = 3
        embed_dim = 4
        num_heads = 2
        head_dim = embed_dim // num_heads

        # Example data: each number is just to track where each value goes
        queries = torch.arange(batch_size * seq_len * embed_dim).view(batch_size, seq_len, embed_dim)
        print("Original tensor (batch_size, seq_len, embed_dim):")
        print(queries)

        # Option 1: Direct view (without transpose)
        direct_view = queries.view(batch_size, num_heads, seq_len, head_dim)
        print("\nDirect view without transpose (incorrect alignment):")
        print(direct_view)

        # Option 2: Correct view with transpose
        correct_view = queries.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        print("\nCorrect view with transpose (correct alignment):")
        print(correct_view)
        """

        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # calculate self attention
        # query * keys
        # keys.transpose(2, 3) = [batch_size, self.num_heads, self.head_dim, seq_len]
        # so [batch_size, self.num_heads, seq_len, self.head_dim] x [batch_size, self.num_heads,  self.head_dim, seq_len]
        # gives: [batch_size, self.num_heads, seq_len, seq_len]
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be: {(batch_size, self.num_heads, seq_len, seq_len)}"
                + f"\n but is: {attn_weights.size()}"
            )

        # in each element of the batch, we get a matrix like this (example):
        # torch.rand(16).view(4, 4)
        # <------------- seq_len ------------->
        # [
        #   [0.9166, 0.0398, 0.1608, 0.1432 ...],   ↑
        #   [0.8068, 0.5731, 0.2870, 0.1439 ...],   |
        #   [0.5343, 0.2806, 0.7546, 0.5263 ...],   |  seq_len
        #   [0.2353, 0.3923, 0.1000, 0.4808 ...],   |
        #   [ ...  ,   ... ,  ...  ,  ...   ...],   ↓
        # ]
        # along each row, we want probabilities
        # shape of each batch item is [self.num_heads, seq_len, seq_len]
        # we want to apply it ALONG the very last dimension. see this to understand dim vs tensor.shape
        # https://chatgpt.com/share/e/66fad6c1-70b4-800a-98b1-1ccbb0c7c196
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # dropout
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # [batch_size, self.num_heads, seq_len, seq_len] * [batch_size, self.num_heads, seq_len, self.head_dim]
        # gives [batch_size, self.num_heads, seq_len, self.head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # CONCAT heads
        """
        x = torch.arange(16).view(2, 2, 4, 1)
        # batch_size = 2
        # num_heads = 2
        # seq_len = 4
        # head_dim = 1
        # an element in the batch looks like this spread across 2 heads:
        [
            # first head: 4 seq length, 1 dim
            [[ 0],[ 1],[ 2],[ 3]],
            [[ 4],[ 5],[ 6],[ 7]],
        ]
        # first, we want:
        [
            [
                [[ 0],[ 4]],
                [[ 1],[ 5]],
                [[ 2],[ 6]],
                [[ 3],[ 7]]
            ],
        ]
        # which is basically a transpose
        """
        # first transpose and convert to [batch_size, seq_len, self.num_heads, self.head_dim]
        # aka [batch_size, num_patches, self.num_heads, self.head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # now we want to merge all the outputs of the heads into one list like:
        """
        [
            [[ 0, 4]],
            [[ 1, 5]],
            [[ 2, 6]],
            [[ 3, 7]]
        ]
        """
        # why can't we do squeeze?
        # attn_output = attn_output.squeeze(-1)

        # we get [batch_size, num_patches, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # so far, all our heads have worked independently on head_dim
        # but we want to now mix their results
        # so we multiply by a w matrix
        # we get [batch_size, num_patches, embed_dim]
        # this is also what we started with... wtf
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        # b, num_patches, embed_dim -> b, num_patches, intermediate_size
        fc1_output = self.fc1(hidden_states)

        non_lin = nn.functional.gelu(fc1_output, approximate="tanh")
        # b, num_patches, intermediate_size -> b, num_patches, embed_dim
        fc2_output = self.fc2(non_lin)

        return fc2_output


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # this is the first thing after embeddings
        # receives b, num_patches, embed_dim
        # outputs are split into wq, wk and wv
        # also copied for residual connection
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        # this is after layer norm 1 above
        # inputs are wq, wk and wv
        # outputs b, num_patches, embed_dim
        self.self_attn = SiglipAttention(config)

        # after self_attn + self.lyaer_norm1
        # output send to MLP layer
        # also copied for residual connection
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor):
        # b, n_patches, embed_dim
        residual = hidden_states
        # b, n_patches, embed_dim -> b, n_patches, embed_dim
        hidden_states = self.layer_norm1(hidden_states)
        # b, n_patches, embed_dim -> b, n_patches, embed_dim
        hidden_states, _ = self.self_attn(hidden_states)
        # b, n_patches, embed_dim
        hidden_states = residual + hidden_states
        residual = hidden_states
        # b, n_patches, embed_dim -> b, n_patches, embed_dim
        hidden_states = self.layer_norm2(hidden_states)
        # b, n_patches, embed_dim -> b, n_patches, embed_dim
        hidden_states = self.mlp(hidden_states)
        # b, n_patches, embed_dim
        hidden_states = hidden_states + residual

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config=SiglipVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim]
        hidden_states = input_embeds

        for encoder_layer in self.layers:
            # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values = [B, C, H, W] -> [B, Num_patches, Embed_dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(input_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch size, channel, height, width] ->
        # [Batch size, num_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)
