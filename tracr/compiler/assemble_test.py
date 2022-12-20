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
"""Tests for transformer.assemble."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tracr.compiler import assemble
from tracr.craft import bases


class AssembleTest(parameterized.TestCase):

  def test_token_embedding_produces_correct_embedding(self):
    # Token embeddings should be one-hot embeddings of the input integers
    # into the token subspace of residual_space
    input_space = bases.VectorSpaceWithBasis.from_values("0inp", range(2))
    indices_space = bases.VectorSpaceWithBasis.from_values("1ind", range(3))
    output_space = bases.VectorSpaceWithBasis.from_values("2out", range(2))
    residual_space = bases.join_vector_spaces(input_space, indices_space,
                                              output_space)

    @hk.without_apply_rng
    @hk.transform
    def token_pos_embed(tokens):
      embed_modules = assemble._make_embedding_modules(
          residual_space=residual_space,
          tokens_space=input_space,
          indices_space=indices_space,
          output_space=output_space)
      return embed_modules.token_embed(tokens)

    tokens = jnp.array([0, 0, 1])
    expected_token_embeddings = jnp.array([[1, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0, 0]])

    params = token_pos_embed.init(jax.random.PRNGKey(0), tokens)
    embeddings = token_pos_embed.apply(params, tokens)
    np.testing.assert_allclose(embeddings, expected_token_embeddings)

  def test_position_embedding_produces_correct_embedding(self):
    # Position embeddings should be one-hot embeddings of the input integers
    # (representing indices) into the indices subspace of residual_space
    input_space = bases.VectorSpaceWithBasis.from_values("0inp", range(2))
    indices_space = bases.VectorSpaceWithBasis.from_values("1ind", range(3))
    output_space = bases.VectorSpaceWithBasis.from_values("2out", range(2))
    residual_space = bases.join_vector_spaces(input_space, indices_space,
                                              output_space)

    @hk.without_apply_rng
    @hk.transform
    def token_pos_embed(tokens):
      embed_modules = assemble._make_embedding_modules(
          residual_space=residual_space,
          tokens_space=input_space,
          indices_space=indices_space,
          output_space=output_space)
      return embed_modules.pos_embed(jnp.indices(tokens.shape)[-1])

    tokens = jnp.array([3, 0, 0, 1])
    expected_pos_embeddings = jnp.array([[0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 0, 1, 0, 0]])

    params = token_pos_embed.init(jax.random.PRNGKey(0), tokens)
    embeddings = token_pos_embed.apply(params, tokens)
    np.testing.assert_allclose(embeddings, expected_pos_embeddings)

  def test_unembedding(self):
    # Prepend numbers to preserve basis order [input, index, output]
    input_space = bases.VectorSpaceWithBasis.from_values("0inp", range(2))
    indices_space = bases.VectorSpaceWithBasis.from_values("1ind", range(3))
    output_space = bases.VectorSpaceWithBasis.from_values("2out", range(2))
    residual_space = bases.join_vector_spaces(input_space, indices_space,
                                              output_space)

    @hk.without_apply_rng
    @hk.transform
    def unembed(embeddings):
      embed_modules = assemble._make_embedding_modules(
          residual_space=residual_space,
          tokens_space=input_space,
          indices_space=indices_space,
          output_space=output_space)
      return embed_modules.unembed(embeddings, use_unembed_argmax=True)

    embeddings = jnp.array([
        # pylint: disable=g-no-space-after-comment
        #inp| indices| out | < spaces
        #0  1  0  1  2  0  1  < values in spaces
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    expected_tokens = jnp.array([1, 0, 1])

    params = unembed.init(jax.random.PRNGKey(0), embeddings)
    tokens = unembed.apply(params, embeddings)
    np.testing.assert_allclose(tokens, expected_tokens)


if __name__ == "__main__":
  absltest.main()
