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
from tracr.transformer import compressed_model
from tracr.transformer import model


class CompressedTransformerTest(parameterized.TestCase):

  def _check_layer_naming(self, params):
    # Modules should be named for example
    # For MLPs: "compressed_transformer/layer_{i}/mlp/linear_1"
    # For Attention: "compressed_transformer/layer_{i}/attn/key"
    # For Layer Norm: "compressed_transformer/layer_{i}/layer_norm"
    for key in params.keys():
      levels = key.split("/")
      self.assertEqual(levels[0], "compressed_transformer")
      if len(levels) == 1:
        self.assertEqual(list(params[key].keys()), ["w_emb"])
        continue
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
      transformer = compressed_model.CompressedTransformer(
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
      transformer = compressed_model.CompressedTransformer(
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
      transformer = compressed_model.CompressedTransformer(
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
      transformer = compressed_model.CompressedTransformer(
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

  def test_not_setting_embedding_size_produces_same_output_as_default_model(
      self):
    config = model.TransformerConfig(
        num_heads=2,
        num_layers=2,
        key_size=5,
        mlp_hidden_size=64,
        dropout_rate=0.,
        causal=False,
        layer_norm=False)

    @hk.without_apply_rng
    @hk.transform
    def forward_model(emb, mask):
      return model.Transformer(config)(emb, mask).output

    @hk.without_apply_rng
    @hk.transform
    def forward_superposition(emb, mask):
      return compressed_model.CompressedTransformer(config)(emb, mask).output

    seq_len = 4
    emb = np.random.random((1, seq_len, 1))
    mask = np.ones((1, seq_len))
    emb, mask = jnp.array(emb), jnp.array(mask)

    rng = hk.PRNGSequence(1)
    params = forward_model.init(next(rng), emb, mask)
    params_superposition = {
        k.replace("transformer", "compressed_transformer"): v
        for k, v in params.items()
    }

    out_model = forward_model.apply(params, emb, mask)
    out_superposition = forward_superposition.apply(params_superposition, emb,
                                                    mask)

    self._check_layer_naming(params_superposition)
    np.testing.assert_allclose(out_model, out_superposition)

  @parameterized.parameters(
      dict(embedding_size=2, unembed_at_every_layer=True),
      dict(embedding_size=2, unembed_at_every_layer=False),
      dict(embedding_size=6, unembed_at_every_layer=True),
      dict(embedding_size=6, unembed_at_every_layer=False))
  def test_embbeding_size_produces_correct_shape_of_residuals_and_layer_outputs(
      self, embedding_size, unembed_at_every_layer):

    @hk.transform
    def forward(emb, mask):
      transformer = compressed_model.CompressedTransformer(
          model.TransformerConfig(
              num_heads=2,
              num_layers=2,
              key_size=5,
              mlp_hidden_size=64,
              dropout_rate=0.,
              causal=False,
              layer_norm=False))
      return transformer(
          emb,
          mask,
          embedding_size=embedding_size,
          unembed_at_every_layer=unembed_at_every_layer,
      )

    seq_len = 4
    model_size = 16

    emb = np.random.random((1, seq_len, model_size))
    mask = np.ones((1, seq_len))
    emb, mask = jnp.array(emb), jnp.array(mask)

    rng = hk.PRNGSequence(1)
    params = forward.init(next(rng), emb, mask)
    activations = forward.apply(params, next(rng), emb, mask)

    self._check_layer_naming(params)

    for residual in activations.residuals:
      self.assertEqual(residual.shape, (1, seq_len, embedding_size))

    for layer_output in activations.layer_outputs:
      self.assertEqual(layer_output.shape, (1, seq_len, model_size))

  @parameterized.parameters(
      dict(model_size=2, unembed_at_every_layer=True),
      dict(model_size=2, unembed_at_every_layer=False),
      dict(model_size=6, unembed_at_every_layer=True),
      dict(model_size=6, unembed_at_every_layer=False))
  def test_identity_embedding_produces_same_output_as_standard_model(
      self, model_size, unembed_at_every_layer):

    config = model.TransformerConfig(
        num_heads=2,
        num_layers=2,
        key_size=5,
        mlp_hidden_size=64,
        dropout_rate=0.,
        causal=False,
        layer_norm=False)

    @hk.without_apply_rng
    @hk.transform
    def forward_model(emb, mask):
      return model.Transformer(config)(emb, mask).output

    @hk.without_apply_rng
    @hk.transform
    def forward_superposition(emb, mask):
      return compressed_model.CompressedTransformer(config)(
          emb,
          mask,
          embedding_size=model_size,
          unembed_at_every_layer=unembed_at_every_layer).output

    seq_len = 4
    emb = np.random.random((1, seq_len, model_size))
    mask = np.ones((1, seq_len))
    emb, mask = jnp.array(emb), jnp.array(mask)

    rng = hk.PRNGSequence(1)
    params = forward_model.init(next(rng), emb, mask)
    params_superposition = {
        k.replace("transformer", "compressed_transformer"): v
        for k, v in params.items()
    }
    params_superposition["compressed_transformer"] = {
        "w_emb": jnp.identity(model_size)
    }

    out_model = forward_model.apply(params, emb, mask)
    out_superposition = forward_superposition.apply(params_superposition, emb,
                                                    mask)

    self._check_layer_naming(params_superposition)
    np.testing.assert_allclose(out_model, out_superposition)


if __name__ == "__main__":
  absltest.main()
