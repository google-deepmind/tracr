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
"""Tests for transformer.model."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tracr.transformer import model


class TransformerTest(parameterized.TestCase):

  def _check_layer_naming(self, params):
    # Modules should be named for example
    # For MLPs: "transformer/layer_{i}/mlp/linear_1"
    # For Attention: "transformer/layer_{i}/attn/key"
    # For Layer Norm: "transformer/layer_{i}/layer_norm"
    for key in params.keys():
      levels = key.split("/")
      self.assertEqual(levels[0], "transformer")
      if levels[1].startswith("layer_norm"):
        continue  # output layer norm
      self.assertStartsWith(levels[1], "layer")
      if levels[2] == "mlp":
        self.assertIn(levels[3], {"linear_1", "linear_2"})
      elif levels[2] == "attn":
        self.assertIn(levels[3], {"key", "query", "value", "linear"})
      else:
        self.assertStartsWith(levels[2], "layer_norm")

  def _zero_mlps(self, params):
    for module in params:
      if "mlp" in module:
        for param in params[module]:
          params[module][param] = jnp.zeros_like(params[module][param])
    return params

  @parameterized.parameters(dict(layer_norm=True), dict(layer_norm=False))
  def test_layer_norm(self, layer_norm):
    # input = [1, 1, 1, 1]
    # If layer norm is used, this should give all-0 output for a freshly
    # initialized model because LN will subtract the mean after each layer.
    # Else we expect non-zero outputs.

    @hk.transform
    def forward(emb, mask):
      transformer = model.Transformer(
          model.TransformerConfig(
              num_heads=2,
              num_layers=2,
              key_size=5,
              mlp_hidden_size=64,
              dropout_rate=0.,
              layer_norm=layer_norm))
      return transformer(emb, mask).output

    seq_len = 4
    emb = jnp.ones((1, seq_len, 1))
    mask = jnp.ones((1, seq_len))
    rng = hk.PRNGSequence(1)
    params = forward.init(next(rng), emb, mask)
    out = forward.apply(params, next(rng), emb, mask)

    self._check_layer_naming(params)
    if layer_norm:
      np.testing.assert_allclose(out, 0)
    else:
      self.assertFalse(np.allclose(out, 0))

  @parameterized.parameters(dict(causal=True), dict(causal=False))
  def test_causal_attention(self, causal):
    # input = [0, random, random, random]
    # mask = [1, 0, 1, 1]
    # For causal attention the second token can only attend to the first one, so
    # it should be the same. For non-causal attention all tokens should change.

    @hk.transform
    def forward(emb, mask):
      transformer = model.Transformer(
          model.TransformerConfig(
              num_heads=2,
              num_layers=2,
              key_size=5,
              mlp_hidden_size=64,
              dropout_rate=0.,
              layer_norm=False,
              causal=causal))
      return transformer(emb, mask).output

    seq_len = 4
    emb = np.random.random((1, seq_len, 1))
    emb[:, 0, :] = 0
    mask = np.array([[1, 0, 1, 1]])
    emb, mask = jnp.array(emb), jnp.array(mask)

    rng = hk.PRNGSequence(1)
    params = forward.init(next(rng), emb, mask)
    params = self._zero_mlps(params)
    out = forward.apply(params, next(rng), emb, mask)

    self._check_layer_naming(params)
    if causal:
      self.assertEqual(0, out[0, 0, 0])
      self.assertEqual(emb[0, 1, 0], out[0, 1, 0])
    else:
      self.assertNotEqual(0, out[0, 0, 0])
      self.assertNotEqual(emb[0, 1, 0], out[0, 1, 0])
    self.assertNotEqual(emb[0, 2, 0], out[0, 2, 0])
    self.assertNotEqual(emb[0, 3, 0], out[0, 3, 0])

  def test_setting_activation_function_to_zero(self):
    # An activation function that always returns zeros should result in the
    # same model output as setting all MLP weights to zero.

    @hk.transform
    def forward_zero(emb, mask):
      transformer = model.Transformer(
          model.TransformerConfig(
              num_heads=2,
              num_layers=2,
              key_size=5,
              mlp_hidden_size=64,
              dropout_rate=0.,
              causal=False,
              layer_norm=False,
              activation_function=jnp.zeros_like))
      return transformer(emb, mask).output

    @hk.transform
    def forward(emb, mask):
      transformer = model.Transformer(
          model.TransformerConfig(
              num_heads=2,
              num_layers=2,
              key_size=5,
              mlp_hidden_size=64,
              dropout_rate=0.,
              causal=False,
              layer_norm=False,
              activation_function=jax.nn.gelu))
      return transformer(emb, mask).output

    seq_len = 4
    emb = np.random.random((1, seq_len, 1))
    mask = np.ones((1, seq_len))
    emb, mask = jnp.array(emb), jnp.array(mask)

    rng = hk.PRNGSequence(1)
    params = forward.init(next(rng), emb, mask)
    params_no_mlps = self._zero_mlps(params)

    out_zero_activation = forward_zero.apply(params, next(rng), emb, mask)
    out_no_mlps = forward.apply(params_no_mlps, next(rng), emb, mask)

    self._check_layer_naming(params)
    np.testing.assert_allclose(out_zero_activation, out_no_mlps)
    self.assertFalse(np.allclose(out_zero_activation, 0))


class CompiledTransformerModelTest(parameterized.TestCase):

  def _get_one_hot_embed_unembed(self, vocab_size, max_seq_len):
    # Embeds tokens as one-hot into the first `vocab_size` dimensions
    token_embed = hk.Embed(
        embedding_matrix=jnp.block(
            [jnp.eye(vocab_size),
             jnp.zeros((vocab_size, max_seq_len))]))

    # Embeds positions as one-hot into the last `max_seq_len` dimensions
    position_embed = hk.Embed(
        embedding_matrix=jnp.block(
            [jnp.zeros((max_seq_len, vocab_size)),
             jnp.eye(max_seq_len)]))

    class Unembed(hk.Module):

      def __call__(self, embeddings):
        return jnp.argmax(embeddings[:, :, :vocab_size], axis=-1)

    return token_embed, position_embed, Unembed()

  def test_embedding_gives_desired_result(self):
    tokens = jnp.array([[1, 2, 3]])
    vocab_size, max_seq_len, pad_token = 5, 5, 0

    expected_embeddings = jnp.array([[[0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]]])

    @hk.transform
    def embed(tokens):
      transformer = model.Transformer(
          model.TransformerConfig(
              num_heads=2,
              num_layers=2,
              key_size=5,
              mlp_hidden_size=64,
              dropout_rate=0.,
              causal=False,
              layer_norm=False,
              activation_function=jax.nn.gelu))
      token_embed, position_embed, unembed = self._get_one_hot_embed_unembed(
          vocab_size, max_seq_len)
      compiled_model = model.CompiledTransformerModel(
          transformer=transformer,
          token_embed=token_embed,
          position_embed=position_embed,
          unembed=unembed,
          use_unembed_argmax=True,
          pad_token=pad_token)
      return compiled_model.embed(tokens)

    rng = hk.PRNGSequence(1)
    params = embed.init(next(rng), tokens)
    embeddings = embed.apply(params, next(rng), tokens)

    np.testing.assert_allclose(embeddings, expected_embeddings)

  def test_embedding_then_unembedding_gives_same_tokens(self):
    tokens = jnp.array([[1, 2, 3], [4, 5, 6], [3, 2, 4]])
    vocab_size, max_seq_len, pad_token = 10, 5, 0

    @hk.transform
    def embed_unembed(tokens):
      transformer = model.Transformer(
          model.TransformerConfig(
              num_heads=2,
              num_layers=2,
              key_size=5,
              mlp_hidden_size=64,
              dropout_rate=0.,
              causal=False,
              layer_norm=False,
              activation_function=jax.nn.gelu))
      token_embed, position_embed, unembed = self._get_one_hot_embed_unembed(
          vocab_size, max_seq_len)
      compiled_model = model.CompiledTransformerModel(
          transformer=transformer,
          token_embed=token_embed,
          position_embed=position_embed,
          unembed=unembed,
          use_unembed_argmax=True,
          pad_token=pad_token)
      embeddings = compiled_model.embed(tokens)
      unembeddings = compiled_model.unembed(embeddings)
      return embeddings, unembeddings

    rng = hk.PRNGSequence(1)
    params = embed_unembed.init(next(rng), tokens)
    embeddings, unembeddings = embed_unembed.apply(params, next(rng), tokens)

    self.assertEqual(
        embeddings.shape,
        (tokens.shape[0], tokens.shape[1], vocab_size + max_seq_len))

    np.testing.assert_allclose(unembeddings, tokens)


if __name__ == "__main__":
  absltest.main()
