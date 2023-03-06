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
"""Assemble weights of a transformer model from a craft residual stack."""

import dataclasses
from typing import Any, Callable, Optional, List, Tuple

import chex
import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tracr.craft import bases
from tracr.craft import transformers
from tracr.craft import vectorspace_fns
from tracr.transformer import encoder
from tracr.transformer import model
from typing_extensions import Protocol


@chex.dataclass
class AssembledTransformerModelOutput:
  decoded: List[Any]  # length T.
  unembedded: jax.Array  # [B, T]     B = 1 always.
  layer_outputs: List[jax.Array]  # [B, T, D]
  residuals: List[jax.Array]  # [B, T, D]
  attn_logits: List[jax.Array]  # [B, T, T, H]
  transformer_output: jax.Array  # [B, T, D]
  input_embeddings: jax.Array


class ModelForward(Protocol):

  def __call__(
      self,
      params: hk.Params,
      emb: jax.Array,
  ) -> model.CompiledTransformerModelOutput:
    """A hk-transformed forward pass through the compiled model."""


@dataclasses.dataclass
class AssembledTransformerModel:
  """Model architecture and parameters from assembling a model."""
  forward: ModelForward
  get_compiled_model: Callable[[], model.CompiledTransformerModel]
  params: hk.Params
  model_config: model.TransformerConfig
  residual_labels: List[str]
  input_encoder: Optional[encoder.Encoder] = None
  output_encoder: Optional[encoder.Encoder] = None

  def apply(self, tokens: List[bases.Value]) -> AssembledTransformerModelOutput:
    """Returns output from running the model on a set of input tokens."""
    if self.input_encoder:
      tokens = self.input_encoder.encode(tokens)
    tokens = jnp.array([tokens])
    output = self.forward(self.params, tokens)
    decoded = output.unembedded_output[0].tolist()
    if self.output_encoder:
      decoded = self.output_encoder.decode(decoded)

    if self.input_encoder.bos_token:
      # Special case for decoding the bos token position, for which the output
      # decoder might have unspecified behavior.
      decoded = [self.input_encoder.bos_token] + decoded[1:]

    return AssembledTransformerModelOutput(
        decoded=decoded,
        unembedded=output.unembedded_output,
        layer_outputs=output.transformer_output.layer_outputs,
        residuals=output.transformer_output.residuals,
        attn_logits=output.transformer_output.attn_logits,
        transformer_output=output.transformer_output.output,
        input_embeddings=output.transformer_output.input_embeddings)


@dataclasses.dataclass
class EmbeddingModules:
  """Modules for embedding and tokens and positions and unembedding results."""
  token_embed: model.CallableHaikuModule
  pos_embed: model.CallableHaikuModule
  unembed: model.CallableHaikuModule


def _get_model_config_and_module_names(
    craft_model: transformers.SeriesWithResiduals
) -> Tuple[model.TransformerConfig, List[str]]:
  """Returns model config and locations (in params) for halflayers."""

  multi_attn_heads: List[List[transformers.AttentionHead]] = []
  mlps: List[transformers.MLP] = []
  module_names: List[str] = []

  candidate_module_names = []
  for layer in range(len(craft_model.blocks)):
    candidate_module_names.append(f"transformer/layer_{layer}/attn")
    candidate_module_names.append(f"transformer/layer_{layer}/mlp")
  candidate_module_names = iter(candidate_module_names)

  for module in craft_model.blocks:
    if isinstance(module, transformers.MLP):
      mlps.append(module)
      layer_type = "mlp"
    else:
      multi_attn_heads.append(list(module.as_multi().heads()))
      layer_type = "attn"
    # Find next layer with the necessary type. Modules in-between, that are not
    # added to module_names will be disabled later by setting all weights to 0.
    module_name = next(candidate_module_names)
    while layer_type not in module_name:
      module_name = next(candidate_module_names)
    module_names.append(module_name)

  num_layers = int(module_names[-1].split("_")[1].split("/")[0]) + 1
  heads = sum(multi_attn_heads, [])

  if multi_attn_heads:
    num_heads = max(len(heads) for heads in multi_attn_heads)
    key_size = max(max(head.w_qk.matrix.shape) for head in heads)
  else:
    num_heads, key_size = 1, 1

  if mlps:
    mlp_hidden_size = max(mlp.fst.output_space.num_dims for mlp in mlps)
  else:
    mlp_hidden_size = 1

  model_config = model.TransformerConfig(
      num_heads=num_heads,
      num_layers=num_layers,
      key_size=key_size,
      mlp_hidden_size=mlp_hidden_size,
      dropout_rate=0.,
      activation_function=jax.nn.relu,
      layer_norm=False,
      causal=False,
  )

  return model_config, module_names


def _make_embedding_modules(
    residual_space: bases.VectorSpaceWithBasis,
    tokens_space: bases.VectorSpaceWithBasis,
    indices_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis) -> EmbeddingModules:
  """Creates embedding and unembedding modules from vector spaces.

  Args:
    residual_space: Full residual space of the model.
    tokens_space: Subspace to embed tokens to.
    indices_space: Subspace to embed indices/position embeddings to.
    output_space: Subspace to unembed outputs from.

  Returns:
    EmbeddingModules containing modules for token embeddings, position
      embeddings and unembeddings.
  """
  tokens_to_res = vectorspace_fns.project(tokens_space, residual_space)

  # If we use the 'one' direction, make sure all inputs have a 1 here
  one_dir = bases.BasisDirection("one")
  if one_dir in residual_space:
    one_to_res = vectorspace_fns.Linear.from_action(
        tokens_space, residual_space,
        lambda x: residual_space.vector_from_basis_direction(one_dir))
    tokens_to_res = vectorspace_fns.Linear.combine_in_parallel(
        [tokens_to_res, one_to_res])

  # Token embeddings.
  res_to_out = vectorspace_fns.project(residual_space, output_space)
  token_embed = hk.Embed(  # pytype: disable=wrong-arg-types  # jax-ndarray
      embedding_matrix=tokens_to_res.matrix, name="token_embed")

  # Positional embeddings.
  index_to_res = vectorspace_fns.project(indices_space, residual_space)
  # The zeroth position should not have any positional embeddings,
  # so we add one line of padding at the zeroth position.
  pos_matrix = np.concatenate(
      [np.zeros((1, residual_space.num_dims)), index_to_res.matrix], axis=0)
  pos_embed = hk.Embed(embedding_matrix=pos_matrix, name="pos_embed")

  def unembed(x, use_unembed_argmax):
    out = x @ res_to_out.matrix
    if use_unembed_argmax:
      return jnp.argmax(out, axis=-1)
    elif out.shape[-1] == 1:
      return out.squeeze(-1)
    return out

  unembed_mod = hk.to_module(unembed)()
  return EmbeddingModules(
      token_embed=token_embed, pos_embed=pos_embed, unembed=unembed_mod)


def assemble_craft_model(
    craft_model: transformers.SeriesWithResiduals,
    tokens_space: bases.VectorSpaceWithBasis,
    indices_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis,
    categorical_output: bool,
    causal: bool = False,
) -> AssembledTransformerModel:
  """Assembles the given components into a Haiku model with parameters.

  Args:
    craft_model: Model to assemble weights for.
    tokens_space: Vectorspace to embed the input tokens to.
    indices_space: Vectorspace to embed the indices to (position encodings).
    output_space: Vectorspace that the model will write outputs to that should
      be unembedded.
    categorical_output: Whether the output is categorical. If True, we take an
      argmax when unembedding.
    causal: Whether to output a causally-masked model.

  Returns:
    An AssembledTransformerModel that contains the model and parameters of the
      assembled transformer.
  """
  # TODO(b/255936413): Make embeddings only retain the tokens and indices that
  #   are actually used.
  # TODO(b/255936496): Think about enabling layer norm and reversing it somehow

  model_config, module_names = _get_model_config_and_module_names(craft_model)
  model_config.causal = causal

  residual_space = bases.join_vector_spaces(craft_model.residual_space,
                                            tokens_space, indices_space,
                                            output_space)
  residual_labels = [str(basis_dir) for basis_dir in residual_space.basis]

  # Build model with embedding and unembedding layers
  def get_compiled_model():
    transformer = model.Transformer(model_config)
    embed_modules = _make_embedding_modules(
        residual_space=residual_space,
        tokens_space=tokens_space,
        indices_space=indices_space,
        output_space=output_space)
    return model.CompiledTransformerModel(
        transformer=transformer,
        token_embed=embed_modules.token_embed,
        position_embed=embed_modules.pos_embed,
        unembed=embed_modules.unembed,
        use_unembed_argmax=categorical_output)

  @hk.without_apply_rng
  @hk.transform
  def forward(emb):
    compiled_model = get_compiled_model()
    return compiled_model(emb, use_dropout=False)

  params = forward.init(jax.random.PRNGKey(0), jnp.array([[1, 2, 3]]))
  params = {k: dict(v) for k, v in params.items()}

  for key in params:
    if "transformer" in key:
      for par in params[key]:
        params[key][par] = np.zeros_like(params[key][par])

  # Assemble attention and MLP weights.
  project = lambda space: vectorspace_fns.project(residual_space, space).matrix

  for module_name, module in zip(module_names, craft_model.blocks):
    if isinstance(module, transformers.MLP):
      hidden_size = module.fst.output_space.num_dims
      residual_to_fst_input = project(module.fst.input_space)
      snd_output_to_residual = project(module.snd.output_space).T
      params[f"{module_name}/linear_1"]["w"][:, :hidden_size] = (
          residual_to_fst_input @ module.fst.matrix)
      params[f"{module_name}/linear_2"]["w"][:hidden_size, :] = (
          module.snd.matrix @ snd_output_to_residual)
    else:  # Attention module
      query, key, value, linear = [], [], [], []
      for head in module.as_multi().heads():
        key_size = head.w_qk.matrix.shape[1]
        query_mat = np.zeros((residual_space.num_dims, model_config.key_size))
        residual_to_query = project(head.w_qk.left_space)
        query_mat[:, :key_size] = residual_to_query @ head.w_qk.matrix
        query.append(query_mat)

        key_mat = np.zeros((residual_space.num_dims, model_config.key_size))
        key_mat[:, :key_size] = project(head.w_qk.right_space)
        key.append(key_mat)

        value_size = head.w_ov.matrix.shape[1]
        value_mat = np.zeros((residual_space.num_dims, model_config.key_size))
        residual_to_ov_input = project(head.w_ov.input_space)
        value_mat[:, :value_size] = residual_to_ov_input @ head.w_ov.matrix
        value.append(value_mat)

        linear_mat = np.zeros((model_config.key_size, residual_space.num_dims))
        linear_mat[:value_size, :] = project(head.w_ov.output_space).T
        linear.append(linear_mat)

      # Fill up heads that are not used with zero weights
      for _ in range(model_config.num_heads - module.as_multi().num_heads):
        query.append(np.zeros_like(query[0]))
        key.append(np.zeros_like(key[0]))
        value.append(np.zeros_like(value[0]))
        linear.append(np.zeros_like(linear[0]))

      query = einops.rearrange(query,
                               "heads input output -> input (heads output)")
      key = einops.rearrange(key, "heads input output -> input (heads output)")
      value = einops.rearrange(value,
                               "heads input output -> input (heads output)")
      linear = einops.rearrange(linear,
                                "heads input output -> (heads input) output")

      params[f"{module_name}/query"]["w"][:, :] = query
      params[f"{module_name}/key"]["w"][:, :] = key
      params[f"{module_name}/value"]["w"][:, :] = value
      params[f"{module_name}/linear"]["w"][:, :] = linear

  params = jax.tree_util.tree_map(jnp.array, params)
  return AssembledTransformerModel(
      forward=forward.apply,
      get_compiled_model=get_compiled_model,
      params=params,
      model_config=model_config,
      residual_labels=residual_labels,
  )
