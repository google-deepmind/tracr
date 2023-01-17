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
"""Attention head for categorical inputs."""

from typing import Optional

from tracr.craft import bases
from tracr.craft import transformers
from tracr.craft import vectorspace_fns
from typing_extensions import Protocol


class QueryKeyToAttnLogit(Protocol):

  def __call__(self, query: bases.BasisDirection,
               key: bases.BasisDirection) -> bool:
    pass


def categorical_attn(
    query_space: bases.VectorSpaceWithBasis,
    key_space: bases.VectorSpaceWithBasis,
    value_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis,
    bos_space: bases.VectorSpaceWithBasis,
    one_space: bases.VectorSpaceWithBasis,
    attn_fn: QueryKeyToAttnLogit,
    default_output: Optional[bases.VectorInBasis] = None,
    causal: bool = False,
    always_attend_to_bos: bool = False,
    use_bos_for_default_output: bool = True,
    softmax_coldness: float = 100.,
) -> transformers.AttentionHead:
  """Returns an attention head for categorical inputs.

  Assumes the existence of a beginning of sequence token and attends to it
  always with strength 0.5*softmax_coldness. This allows to implement an
  arbitrary default value for rows in the attention pattern that are all-zero.

  Attends to the BOS token if all other key-query pairs have zero attention.
  Hence, the first value in the value sequence will be the default output for
  such cases.

  Args:
    query_space: Vector space containing (categorical) query input.
    key_space: Vector space containing (categorical) key input.
    value_space: Vector space containing (numerical) value input.
    output_space: Vector space which will contain (numerical) output.
    bos_space: 1-d space used to identify the beginning of sequence token.
    one_space: 1-d space which contains 1 at every position.
    attn_fn: A selector function f(query, key) operating on the query/key basis
      directions that defines the attention pattern.
    default_output: Output to return if attention pattern is all zero.
    causal: If True, use masked attention.
    always_attend_to_bos: If True, always attend to the BOS token. If False,
      only attend to BOS when attending to nothing else.
    use_bos_for_default_output: If True, assume BOS is not in the value space
      and output a default value when attending to BOS. If False, assume BOS is
      in the value space, and map it to the output space like any other token.
    softmax_coldness: The inverse temperature of the softmax. Default value is
      high which makes the attention close to a hard maximum.
  """
  bases.ensure_dims(bos_space, num_dims=1, name="bos_space")
  bases.ensure_dims(one_space, num_dims=1, name="one_space")
  bos_direction = bos_space.basis[0]
  one_direction = one_space.basis[0]

  # Add bos direction to query, key, and value spaces in case it is missing
  query_space = bases.join_vector_spaces(query_space, bos_space, one_space)
  key_space = bases.join_vector_spaces(key_space, bos_space)
  value_space = bases.join_vector_spaces(value_space, bos_space)

  if always_attend_to_bos:
    value_basis = value_space.basis
  else:
    value_basis = [v for v in value_space.basis if v != bos_direction]
  assert len(value_basis) == output_space.num_dims
  value_to_output = dict(zip(value_basis, output_space.basis))

  if default_output is None:
    default_output = output_space.null_vector()
  assert default_output in output_space

  def qk_fun(query: bases.BasisDirection, key: bases.BasisDirection) -> float:

    # We want to enforce the following property on our attention patterns:
    # - if nothing else is attended to, attend to the BOS token.
    # - otherwise, don't attend to the BOS token.
    #
    # We assume that the BOS position always only contains the vector bos + one,
    # and that any other position has bos coefficient 0.
    #
    # We do this as follows:
    # Let Q and K be subspaces of V containing the query and key vectors,
    # both disjoint with the BOS space {bos} or the one space {one}.
    # Suppose we have an attn_fn which defines a bilinear W_QK: V x V -> ℝ,
    # s.t. W_QK(q, k) = 0 whenever either q or k are bos or one.
    #
    # Then define W_new: V x V -> ℝ st:
    # W_new(one, bos) = 0.5, otherwise 0.
    #
    # Now set W_QK' = W_QK + W_new.
    #
    # To evaluate the attention to the BOS position:
    # W_QK'(q, bos + one)
    # = W_QK'(q, bos) + W_QK'(q, one)
    # = W_QK(q, bos) + W_QK(q, one) + W_new(q, bos) + W_new(q, one)
    # = 0            + 0            + W_new(q, bos) + W_new(q, one)
    # = W_new(q, bos) + W_new(q, one)
    # = W_new(q' + one, bos) + W_new(q' + one, one)  where q = one + q'
    # = W_new(q', bos) + W_new(one, bos) + W_new(q', one) + W_new(one, one)
    # = 0              + 0.5             + 0              + 0
    # = 0.5
    #
    # To evaluate the attention to a non-BOS position:
    # W_QK'(0 * bos + q, 0 * bos + k)  # s.t. q ∈ Q+{one}, k ∈ K+{one}
    # = 0*W_QK'(bos, 0*bos + k) + W_QK'(q, 0*bos + k)
    # = W_QK'(q, 0*bos + k)
    # = 0*W_QK'(q, bos) + W_QK'(q, k)
    # = W_QK'(q, k)
    # = W_QK(q, k)    since W_QK' = W_QK on inputs not containing bos.
    # = W_QK(q', k')  since W_QK(x, y) = 0 whenever x or y are one.
    #
    # Since W_QK(q, k) takes values in 0, 1, a sufficiently high softmax
    # coldness will give us the desired property.                            QED
    #
    # The following implements this idea.
    # By replacing 0.5 with 1, we can instead enforce a different property: that
    # the BOS token is always attended to in addition to whatever else.

    if key == bos_direction and query == one_direction:
      c = 1. if always_attend_to_bos else 0.5
      return c * softmax_coldness
    elif {key, query}.intersection({one_direction, bos_direction}):
      return 0

    return softmax_coldness * attn_fn(query, key)

  w_qk = vectorspace_fns.ScalarBilinear.from_action(
      query_space,
      key_space,
      qk_fun,
  )

  def ov_fun(input_dir: bases.BasisDirection) -> bases.VectorInBasis:
    if use_bos_for_default_output and input_dir == bos_direction:
      return default_output
    return output_space.vector_from_basis_direction(value_to_output[input_dir])

  w_ov = vectorspace_fns.Linear.from_action(
      value_space,
      output_space,
      ov_fun,
  )

  return transformers.AttentionHead(w_qk, w_ov, causal=causal)
