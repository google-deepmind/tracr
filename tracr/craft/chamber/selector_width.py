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
"""SelectorWidth component consisting of an attention head and an MLP."""

from typing import Iterable
from tracr.craft import bases
from tracr.craft import transformers
from tracr.craft import vectorspace_fns
from tracr.craft.chamber import categorical_attn
from tracr.craft.chamber import numerical_mlp


def selector_width(
    query_space: bases.VectorSpaceWithBasis,
    key_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis,
    bos_space: bases.VectorSpaceWithBasis,
    one_space: bases.VectorSpaceWithBasis,
    attn_fn: categorical_attn.QueryKeyToAttnLogit,
    out_value_set: Iterable[float],
    categorical_output: bool,
    causal: bool = False,
    softmax_coldness: float = 100.,
    mlp_large_number: float = 100.,
    label: str = "",
) -> transformers.SeriesWithResiduals:
  """Returns a craft block implementing RASP's SelectorWidth primitive.

  The block consists of one attention head and one MLP.

  The attention head implements the attention pattern (attn_fn or key=bos) and
  aggregates the bos dimension over this pattern. The output of this will be
  1/(d+1) in every position, where d is the "width" of the attention pattern,
  i.e. the number of 1s in a row.

  The MLP then computes d from the previous output in all positions except for
  the first BOS position. In the BOS position the MLP removes the output of the
  attention head, to ensure it only contains the encoding of the BOS token
  which is expected by all other model components.

  Args:
    query_space: Vector space containing (categorical) query input.
    key_space: Vector space containing (categorical) key input.
    output_space: Vector space which will contain (numerical or categorical)
      output.
    bos_space: 1-d space used to identify the beginning of sequence token.
    one_space: Auxiliary 1-d vector space that must contain 1 in the input.
    attn_fn: A selector function f(query, key) operating on the query/key basis
      directions that defines the attention pattern to compute the width of.
    out_value_set: Set of possible output values of this SelectorWidth.
    categorical_output: If True, encode the output as a categorical variable.
    causal: If True, use masked attention.
    softmax_coldness: The inverse temperature of the softmax. Default value is
      high which makes the attention close to a hard maximum.
    mlp_large_number: A larger number makes the MLP more accurate.
    label: A name for this block, used to label auxiliary dimensions.
  """
  assert output_space.num_dims == 1 or categorical_output

  attn_out_dir = bases.BasisDirection(f"{label}_selector_width_attn_output")
  attn_out_space = bases.VectorSpaceWithBasis([attn_out_dir])
  attn_out_vec = attn_out_space.vector_from_basis_direction(attn_out_dir)

  attn = categorical_attn.categorical_attn(
      query_space=query_space,
      key_space=key_space,
      value_space=bos_space,
      output_space=attn_out_space,
      bos_space=bos_space,
      one_space=one_space,
      attn_fn=attn_fn,
      default_output=attn_out_space.null_vector(),
      causal=causal,
      always_attend_to_bos=True,
      use_bos_for_default_output=False,
      softmax_coldness=softmax_coldness)

  fun = lambda x: round((1 / x) - 1)
  in_value_set = {1 / (out_v + 1) for out_v in out_value_set}
  if categorical_output:
    mlp = numerical_mlp.map_numerical_to_categorical_mlp(
        f=fun,
        input_space=attn_out_space,
        output_space=output_space,
        input_value_set=in_value_set,
        one_space=one_space,
        hidden_name=f"_hidden_{label}_",
        large_number=mlp_large_number)
  else:
    mlp = numerical_mlp.map_numerical_mlp(
        f=fun,
        input_space=attn_out_space,
        output_space=output_space,
        input_value_set=in_value_set,
        one_space=one_space,
        hidden_name=f"_hidden_{label}_",
        large_number=mlp_large_number)

  # This implementation of selector width writes at each position including
  # the BOS. To ensure that the BOS token position does not contain
  # additional values, we add an mlp to subtract the output of both layers.
  clean_bos_out_space = bases.join_vector_spaces(attn_out_space, output_space)
  vec_to_subtract_from_bos = attn_out_vec.project(clean_bos_out_space)

  if categorical_output:
    # Add the one-hot encoding of the zero value to the vector
    # which will get scrubbed from the BOS position.
    zero_dir = [d for d in output_space.basis if d.value == 0][0]
    zero_vec = clean_bos_out_space.vector_from_basis_direction(zero_dir)
    vec_to_subtract_from_bos += zero_vec

  # Construct an MLP that subtracts vec_to_subtract_from_bos * bos
  # from the residual stream which is vec_to_subtract_from_bos in the
  # bos position and 0 else. vec_to_subtract_from_bos contains what the
  # attention head writes to the bos position.

  hidden_dir = bases.BasisDirection("_hidden_clean_bos_")
  hidden_space = bases.VectorSpaceWithBasis([hidden_dir])
  hidden_vec = hidden_space.vector_from_basis_direction(hidden_dir)

  # It's okay to use the local variables because they are only used within
  # the same loop iteration to create the MLP.
  # pylint: disable=cell-var-from-loop
  first_layer = vectorspace_fns.Linear.from_action(bos_space, hidden_space,
                                                   lambda x: hidden_vec)
  second_layer = vectorspace_fns.Linear.from_action(
      hidden_space, clean_bos_out_space, lambda x: -vec_to_subtract_from_bos)
  # pylint: enable=cell-var-from-loop
  clean_bos_mlp = transformers.MLP(first_layer, second_layer)

  mlp = transformers.MLP.combine_in_parallel([mlp, clean_bos_mlp])
  return transformers.SeriesWithResiduals([attn, mlp])
