# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Didactic example of an autoregressive Transformer-based language model.

Glossary of shapes:
- B: Batch size.
- T: Sequence length.
- D: Model embedding size.
- H: Number of attention heads.
- V: Vocabulary size.

Forked from: haiku.examples.transformer.model
"""

import collections
import dataclasses
from typing import Callable, List, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tracr.transformer import attention

# hk.Modules are not always callable: github.com/deepmind/dm-haiku/issues/52
# Ideally, we'd want a type:
# CallableHaikuModule = Intersection[Callable[..., jax.Array], hk.Module]
# But Intersection does not exist (yet): github.com/python/typing/issues/213
CallableHaikuModule = Callable[..., jax.Array]


@chex.dataclass
class TransformerOutput:
  layer_outputs: List[jax.Array]  # [B, T, D]
  residuals: List[jax.Array]  # [B, T, D]
  attn_logits: List[jax.Array]  # [B, H, T, T]
  output: jax.Array  # [B, T, D]
  input_embeddings: jax.Array  # [B, T, D]


@dataclasses.dataclass
class TransformerConfig:
  num_heads: int
  num_layers: int
  key_size: int
  mlp_hidden_size: int
  dropout_rate: float
  activation_function: Callable[[jax.Array], jax.Array] = jax.nn.gelu
  layer_norm: bool = True
  causal: bool = False


@dataclasses.dataclass
class Transformer(hk.Module):
  """A transformer stack."""

  config: TransformerConfig
  name: Optional[str] = None

  def __call__(
      self,
      embeddings: jax.Array,  # [B, T, D]
      mask: jax.Array,  # [B, T]
      *,
      use_dropout: bool = True,
  ) -> TransformerOutput:
    """Transforms input embedding sequences to output embedding sequences."""

    def layer_norm(x: jax.Array) -> jax.Array:
      """Applies a unique LayerNorm to x with default settings."""
      if self.config.layer_norm:
        return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
      return x

    initializer = hk.initializers.VarianceScaling(2 / self.config.num_layers)
    dropout_rate = self.config.dropout_rate if use_dropout else 0.
    _, seq_len, model_size = embeddings.shape

    # Compute causal mask for autoregressive sequence modelling.
    mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]
    mask = mask.repeat(seq_len, axis=2)  # [B, H=1, T, T]

    if self.config.causal:
      causal_mask = np.ones((1, 1, seq_len, seq_len))  # [B=1, H=1, T, T]
      causal_mask = np.tril(causal_mask)
      mask = mask * causal_mask  # [B, H=1, T, T]

    # Set up activation collection.
    collected = collections.defaultdict(list)

    def collect(**kwargs):
      for k, v in kwargs.items():
        collected[k].append(v)

    residual = embeddings
    for layer in range(self.config.num_layers):
      with hk.experimental.name_scope(f"layer_{layer}"):
        # First the attention block.
        attn_block = attention.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_size=self.config.key_size,
            model_size=model_size,
            w_init=initializer,
            name="attn")
        attn_in = layer_norm(residual)
        attn_out = attn_block(attn_in, attn_in, attn_in, mask=mask)
        attn_out, attn_logits = attn_out.out, attn_out.logits
        if dropout_rate > 0:
          attn_out = hk.dropout(hk.next_rng_key(), dropout_rate, attn_out)
        residual = residual + attn_out

        collect(
            residuals=residual, layer_outputs=attn_out, attn_logits=attn_logits)

        # Then the dense block.
        with hk.experimental.name_scope("mlp"):
          dense_block = hk.Sequential([
              hk.Linear(
                  self.config.mlp_hidden_size,
                  w_init=initializer,
                  name="linear_1"),
              self.config.activation_function,
              hk.Linear(model_size, w_init=initializer, name="linear_2"),
          ])
        dense_in = layer_norm(residual)
        dense_out = dense_block(dense_in)
        if dropout_rate > 0:
          dense_out = hk.dropout(hk.next_rng_key(), dropout_rate, dense_out)
        residual = residual + dense_out

        collect(residuals=residual, layer_outputs=dense_out)

    return TransformerOutput(
        residuals=collected["residuals"],
        layer_outputs=collected["layer_outputs"],
        attn_logits=collected["attn_logits"],
        output=layer_norm(residual),
        input_embeddings=embeddings,
    )


@chex.dataclass
class CompiledTransformerModelOutput:
  transformer_output: TransformerOutput
  unembedded_output: jax.Array  # [B, T]


@dataclasses.dataclass
class CompiledTransformerModel(hk.Module):
  """A transformer model with one-hot embeddings."""
  transformer: Transformer
  token_embed: CallableHaikuModule
  position_embed: CallableHaikuModule
  unembed: CallableHaikuModule
  use_unembed_argmax: bool
  pad_token: Optional[int] = None

  def embed(self, tokens: jax.Array) -> jax.Array:
    token_embeddings = self.token_embed(tokens)
    positional_embeddings = self.position_embed(jnp.indices(tokens.shape)[-1])
    return token_embeddings + positional_embeddings  # [B, T, D]

  def __call__(
      self,
      tokens: jax.Array,
      use_dropout: bool = True,
  ) -> CompiledTransformerModelOutput:
    """Embed tokens, pass through model, and unembed output."""
    if self.pad_token is None:
      input_mask = jnp.ones_like(tokens)
    else:
      input_mask = (tokens != self.pad_token)
    input_embeddings = self.embed(tokens)

    transformer_output = self.transformer(
        input_embeddings,
        input_mask,
        use_dropout=use_dropout,
    )
    return CompiledTransformerModelOutput(
        transformer_output=transformer_output,
        unembedded_output=self.unembed(
            transformer_output.output,
            use_unembed_argmax=self.use_unembed_argmax,
        ),
    )
