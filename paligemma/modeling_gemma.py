import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


class KVCache:

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # shape of each key cache is [batch_size, num_heads_kv, seq_len, head_dim]
            # returns the seq_len dimension
            return self.key_cache[0].shape[-2]

    def update(
        self, key_states, value_states, layer_idx
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # append to the sequence length that is stored
            # each tensor has shape [batch_size, num_heads_kv, seq_len, head_dim]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )

            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        # number of heads for queries are different for key, value and queries
        # called "grouped query attention"
        # this is number for query
        num_attention_heads,
        # number of heads for key and values
        num_key_value_heads,
        # how many dims each head will work with
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        # output size of the multi modal projector layer
        projection_dim=2048,
        # hidden size = embedding size of the language model
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vision_config = SiglipVisionConfig(**vision_config)

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps):
        super().__init__()
        # this will take in [batch_Size, seq_len, hidden_size aka dim] inputs
        # so this "weight" will be broadcastable across the last dimension of the input
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor):
        # + self.eps is to prevent division by zero
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, input_embeds: torch.Tensor):
        x = self._norm(input_embeds.float())
        x = x * (1.0 + self.weight.float())

        return x.type_as(input_embeds)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # calculate the theta as per theta_i = base ^ (2i / dim) where i = 0, 1, 2.... d // 2
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )

        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    # @torch.no_grad()
    # def forward(
    #     self, x: torch.Tensor, position_ids: torch.Tensor, seq_len: torch.Tensor = None
    # ):
    #     # position_ids shape: ??

    #     # x shape: [batch_size, num_attention_heads, seq_len, head_dim]
    #     self.inv_freq.to(x.device)

    #     # converts to [1, num_attention_heads, 1, seq_len, head_dim]
    #     inv_freq_expanded: torch.Tensor = self.inv_freq[None, :, None]

    #     inv_freq_expanded = inv_freq_expanded.float().expand(
    #         position_ids.shape[0], -1, 1
    #     )

    #     # position_ids expanded: [batch_size, 1, seq_len]
    #     position_ids_expanded = position_ids[:, None, :].float()
    #     device_type = x.device.type
    #     # device_type = (
    #     #     device_type
    #     #     if isinstance(device_type, str) and device_type != "mps"
    #     #     else "cpu"
    #     # )

    #     with torch.autocast(device_type=device_type, enabled=False):
    #         # multiply each theta by the position (which is the argument of sin and cos fns)
    #         # freqs = [batch_Size, head_dim // 2, 1] @ [batch_size, 1, seq_len]
    #         # then transpose gives [batch_size, seq_len, head_dim // 2]
    #         freqs = (
    #             inv_freq_expanded.float() @ position_ids_expanded.float()
    #         ).transpose(1, 2)

    #         # emb = [batch_size, seq_len, head_dim]
    #         emb = torch.cat((freqs, freqs), dim=-1)

    #         cos = emb.cos()
    #         sin = emb.sin()

    # return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        # device_type = (
        #     device_type
        #     if isinstance(device_type, str) and device_type != "mps"
        #     else "cpu"
        # )

        with torch.autocast(device_type="cpu", enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]  # first half of last dimension
    x2 = x[..., x.shape[-1] // 2 :]  # second half of last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim=1,
):
    # add head dim to both cos and sin
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # apply formula
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor):
        # b, seq_len, hidden_size -> # b, seq_len, intermediate_size
        # gate_output = self.gate_proj(hidden_states)

        # # b, seq_len, intermediate_size -> # b, seq_len, intermediate_size
        # non_lin = nn.functional.gelu(gate_output, approximate="tanh")

        # # b, seq_len, hidden_size -> b, seq_len, intermediate_size
        # up_output = self.up_proj(hidden_states)

        # # b, seq_len, intermediate_size * b, seq_len, intermediate_size
        # # ELEMENT WISE MULTIPLICATION
        # mul = non_lin * up_output

        # # b, seq_len, intermediate_size -> b, seq_len, hidden_size
        # output = self.down_proj(mul)

        return self.down_proj(
            nn.functional.gelu(self.gate_proj(hidden_states), approximate="tanh")
            * self.up_proj(hidden_states)
        )


def repeat_kv(input: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, num_kv_heads, seq_len, head_dim = input.shape
    if n_rep == 1:
        # just return because there's no grouping and each query has it's own head
        return input

    # otherwise
    # we use torch.expand
    # we take a tensor that is [batch_Size, num_kv_heads, seq_len, head_dim]
    # we need to repeat the num_kv_heads dimension n_rep times
    # we first squeeze and create it into [batch_size, num_kv_heads, 1, seq_len, head_dim]
    # input = input[:, :, None, :, :]
    # OR
    # input = input.unsqueeze(2)
    # now expand the second dimension so that it is copied n_rep times
    # input = input.expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)

    # OR, one liner for above two lines:
    input = input[:, :, None, :, :].expand(
        batch_size, num_kv_heads, n_rep, seq_len, head_dim
    )

    # now reshape into batch_size, num_heads, seq_len, head_dim
    input = input.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

    return input


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0

        # multi query attn
        # Wq = [1024, 8 * 128] = [1024, 1024]
        # BUT
        # Wk = [1024, 1 * 128] = [1024, 128]
        # and also Wv = [1024, 1 * 128] = [1024, 128]
        # helps save time in gpu copying of keys and values
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()
        # [b, seq_len, hidden_size] -> [batch_size, seq_len, num_heads_q * head_dim]
        query_states: torch.Tensor = self.q_proj(hidden_states)

        # [b, seq_len, hidden_size] -> [batch_size, seq_len, num_heads_k * head_dim]
        key_states: torch.Tensor = self.k_proj(hidden_states)

        # [b, seq_len, hidden_size] -> [batch_size, seq_len, num_heads_v * head_dim]
        value_states: torch.Tensor = self.v_proj(hidden_states)

        # first view each into [batch_size, seq_len, num_heads, head_dim]
        # and then convert it to [batch_size, num_heads, seq_len, head_dim]
        # see ./modeling_siglip's self_Attention for full explanation why we don't do a view directly

        # [batch_size, num_heads, seq_len, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # [batch_size, num_heads, seq_len, head_dim]
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # [batch_size, num_heads, seq_len, head_dim]
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if kv_cache is not None:
            # note that the function parameters key_states, value_states will be one token at a time
            # but what is returned is the full sequence length of all tokens before this
            key_states, value_states = kv_cache.update(
                key_states, value_states, self.layer_idx
            )

        # repeat the key and value states for all queries to see
        # we do naive implementation, without the GPU kernel custom etc etc
        # torch source here: https://pytorch.org/torchtune/stable/_modules/torchtune/modules/attention.html
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # [batch_size, num_heads_q, seq_len_q, seq_len_kv]
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # [batch_size, num_heads_q, seq_len_q, seq_len_kv]
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # dropout
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # [batch_size, num_heads_q, seq_len_q, seq_len_kv] x [batch_size, num_heads, seq_len, head_dim]
        # -> [batch_size, num_heads_q, seq_len_q, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Wrong size after attention calculation.\n"
                + f"Found {attn_output.size()}"
                + f"\nExpected {(batch_size, self.num_heads, seq_len, self.head_dim)}"
            )

        # now convert it to [batch_size, seq_len_q, num_heads_q, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # now we want to merge all the heads into one [batch_size, seq_len_q, num_heads_q * head_dim]
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # now ffn
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        # [b, seq_len, hidden_size]
        hidden_states = self.input_layernorm(hidden_states)

        # [b, seq_len, hidden_size]
        (hidden_states, _) = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        # [b, seq_len, hidden_size]
        hidden_states = residual + hidden_states
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        # [b, seq_len, hidden_size]
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        input_embeds: torch.Tensor,
        kv_cache: Optional[KVCache],
    ):
        # [batch size, seq_len, hidden_size]
        residual = input_embeds
        # [batch size, seq_len, hidden_size]
        hidden_states = input_embeds
        # [batch size, seq_len, hidden_size]
        # normalize
        normalizer = torch.tensor(
            self.config.hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # batch size, seq_len, hidden_size
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # batch size, seq len, hidden_size
        hidden_states = self.norm(hidden_states)

        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, text_config: GemmaConfig):
        super().__init__()
        self.config = text_config
        self.model = GemmaModel(text_config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(
            text_config.hidden_size, text_config.vocab_size, bias=False
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        input_embeddings: torch.FloatTensor,
        kv_cache: Optional[KVCache],
    ) -> Tuple:
        # [batch_sizer, seq_len, hidden_size] -> [batch_sizer, seq_len, hidden_size]
        hidden_states = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeddings,
            kv_cache=kv_cache,
        )

        # [batch_sizer, seq_len, hidden_size] -> [batch_sizer, seq_len, vocab_size]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {"logits": logits}

        if kv_cache is not None:
            # return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data


class PaliGemmaMultiModalProjector(nn.Module):
    """
    Linear layer that converts the incoming image tokens from the
    image model's dim to the text encoder dims
    """

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def forward(self, image_features):
        hidden_states = self.linear(image_features)

        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def tie_weights(self):
        return self.language_model.tie_weights()

    def merge_image_features_with_text_tokens(
        self,
        image_features: torch.Tensor,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        # extract the embed dim for the image features
        _, _, embed_dim = image_features.shape

        batch_size, seq_len = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device

        # scaling. not sure why.
        # this is [batch, num_patches, hidden_size]
        image_features = image_features / (self.config.hidden_size**0.5)

        # combine the embeddings of the image tokens, text tokens and mask padding tokens
        # prepare a set of zeros with the final shape we want
        final_embedding = torch.zeros(
            batch_size, seq_len, embed_dim, dtype=dtype, device=device
        )

        # first create a mask which only has text tokens
        # [batch_size, seq_len]
        text_mask: torch.Tensor = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )

        # a mask which only has image tokens
        # [batch_size, seq_len]
        image_mask: torch.Tensor = input_ids == self.config.image_token_index

        # a mask which only has padding tokens
        # [batch_size, seq_len]
        pad_mask: torch.Tensor = input_ids == self.pad_token_id

        # some torch sorcery incoming
        """
        # [batch, seq_len]
        input_ids = torch.arange(8).reshape(2, 4)
        # tensor([[0, 1, 2, 3],
        #         [4, 5, 6, 7]])
        
        # [batch, seq_len]
        less5_mask = input_ids < 5
        # tensor([[ True,  True,  True,  True],
        #          [ True, False, False, False]])
        
        # This is fine when we're dealing with *input ids*, but what we eventaully want to do is run this mask on *EMBEDDINGS*
        # If we look at input_embeds, the shape would be something like: [batch_size, seq_len, input_ids, embed_dim]
        # the embeddings will replace the individual input_ids with a looong tensor that is (1, embed_dim) shaped.
        # where we are trying to go is [batch_size, seq_len, embed_dim]

        # So what we want to do to our input_ids which is [batch_size, seq_len] is:
        # 1. add a dimension at the end which is embed_dim size
        # 2. Have all values in the internal most (1, embed_dim) vector to be True
        # 3. then we will have a [batch_size, seq_len, embed_dim] mask.

        # first add a dimension
        # this adds a dimension to the last one. "squeezing" those individual Trues and Falses into their own tensors.z
        # [batch, seq_len] -> [batch, seq_len, 1]
        less5_mask = less5_mask.unsqueeze(-1)

        # tensor([[[ True],
        #           [ True],
        #           [ True],
        #           [ True]],

        #           [[ True],
        #           [False],
        #           [False],
        #           [False]]])

        # torch.expand can be used to expand aka "replicate" aka "create copies of" any dimension that is 1.
        # so we will replicate the last dimension above embed_dim times
        # becuase we want all values in the embedding to come through for all matching input_ids
        # -1 = keep it the same

        embed_dim = 2

        # [batch, seq_len, 1] -> [batch, seq_len, embed_dim]
        less5_mask.expand(-1, -1, embed_dim)
        # tensor([[[ True,  True],
        #          [ True,  True],
        #          [ True,  True],
        #          [ True,  True]],
        # 
        #         [[ True,  True],
        #          [False, False],
        #          [False, False],
        #          [False, False]]])
        """

        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        """
        NOTE
        # we can't do torch.where for the image features unfortunately
        # because image mask is [batch, num_images, embed_dim]
        # and NOT [batch, seq_len, embed_dim]
        # we will do something called masked_scatter
        # let's say
        embed_dim = 1
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4, embed_dim)

        # let's say for each of our sequences, the first two tokens are image tokens
        # add -100 to distiguish our image tokens for now for both our sequences

        x[0][0] = torch.tensor([-100])
        x[0][1] = torch.tensor([-100])

        x[1][0] = torch.tensor([-100])
        x[1][1] = torch.tensor([-100])
        
        # [batch_size, seq_len, embed_dim]
        # tensor([[[-100.],
        #          [-100.],
        #          [2.],
        #          [3].],

        #         [[-100.],
        #          [-100.],
        #          [6.],
        #          [7]]])

        image_mask = x == -100
        
        # that means our image_features will look something like: [batch_size, 2, embed_dim]
        # let's fill them with randoms for now

        image_features = torch.rand(2, 2, embed_dim)

        # tensor([[[0.6047],
        #          [0.6122]],
        # 
        #         [[0.3208],
        #          [0.2188]]])

        # try doing: torch.where(image_mask, image_features, x)
        # we get an error: The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 1
        # sadness.

        # Hence, we use another function called masked_scatter: my_tensor.masked_scatter_(mask, source)
        # which: Copies elements from source into my_tensor at positions where the mask is True.
        # our source is image_features
        # our my_tensor is out final embeddings, x in this case.
       
        final_embeds_with_image_tokens_replaced = x.masked_scatter(image_mask, image_features)
        # and finally we get:
        # tensor([[[   0.4882],
        #          [   0.6827],
        #          [   2.0000],
        #          [   3.0000]],
        # 
        #         [[-100.0000],
        #          [-100.0000],
        #          [   6.0000],
        #          [   7.0000]]])

        """

        # Add the text embeddings
        final_embedding = torch.where(text_mask, input_embeds, final_embedding)

        final_embedding = final_embedding.masked_scatter(image_mask, image_features)

        # pad is normal
        final_embedding = torch.where(
            pad_mask, torch.zeros_like(final_embedding), final_embedding
        )

        # add these final_embeddings (of the image + text prompt)
        # to the kv cache
        # [Batch size, seq_len, embed_dim]
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]

        # create the attention mask

        # if kv cache is empty, fill it all
        # note that we only write for inference here, hence our causal mask is 0 for both if and else here
        # but if we were training,
        # https://github.com/huggingface/transformers/blob/ab97a78130f96e72eec609c56cb5f719529ffb9e/src/transformers/models/paligemma/modeling_paligemma.py#L377-L398
        if kv_cache is None or kv_cache.num_items() == 0:
            # don't mask any token, because we are prefilling
            # this only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # since we're generating
            # first ensure query is 1 single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # no masking here either becuase we're only generating 1 token at a time
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # divide this into heads
        # [batch, q_len, kv_len] -> [batch, num_q_heads, q_len, kv_len]
        # mask remains the same across heads
        causal_mask = causal_mask.unsqueeze(1)

        # this is for rotary position embeddings
        if kv_cache is not None and kv_cache.num_items() > 0:
            # NOTE: the attention mask is all 1s
            # we are generating. so only need to apply RoPE to one token: the last one
            # attention mask is [batch_size, seq_len]
            # get the last item in all batches
            # here cumsum will cumulative sum across the seq len dimension
            # so we get the last "id" of the token
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # we're prefilling
            # we will pass all position_ids to rope
            position_ids = (
                (attention_mask.cumsum(-1))
                .masked_fill_((attention_mask == 0), 1)
                .to(device)
            )

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        # this stuff is coming from the processing_paligemma file
        pixel_values: torch.FloatTensor,
        # this is the tokenized prompt with placeholder image tokens
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        kv_cache: Optional[KVCache],
    ):
        # Make sure the input is right-padded
        assert torch.all(attention_mask == 1), "The input cannot be padded"
        # get initial input embeddings for the inpiut
        # this will give us correct embeddings for the "prefix_text" part ofthe prompt
        # but the <image> will be junk
        # we will later replace the <image> token embeddings with the correct once we get from the vision model
        # these are [batch, seq_len, hidden_size]
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # get image token embeddings
        # first we convert the pixel values to image embeddings
        # pixel values are: [batch, channel, h, w]
        # gives [batch, num_patches, image_embed_size]
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype))

        # project from image_embed_size to text_embed_Size using PaliGemmaMultiModalProjector
        # [batch, num_patches, image_embed_size] -> [batch, num_patches, projection_dim]
        image_features = self.multi_modal_projector(selected_image_feature)

        # ensure that the number of image tokens matches the number of

        # now merge the above projected into the input_embeds

        input_embeds, attention_mask, position_ids = (
            self.merge_image_features_with_text_tokens(
                image_features=image_features,
                input_embeds=input_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
        )

        # [batch_size, seq_len]
        outputs = self.language_model(
            input_embeddings=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        return outputs
