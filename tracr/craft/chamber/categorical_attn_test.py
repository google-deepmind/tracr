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
"""Tests for chamber.categorical_attn."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tracr.craft import bases
from tracr.craft import tests_common
from tracr.craft.chamber import categorical_attn


class CategoricalAttnTest(tests_common.VectorFnTestCase):

  @parameterized.parameters([
      dict(causal=False, input_seq=[1, 2, 3, 4, 5], result_seq=[3, 3, 3, 3, 3]),
      dict(
          causal=True,
          input_seq=[1, 2, 3, 4, 5],
          result_seq=[1, 1.5, 2, 2.5, 3]),
      dict(causal=False, input_seq=[10], result_seq=[10]),
      dict(causal=True, input_seq=[10], result_seq=[10]),
      dict(causal=False, input_seq=[-1, 0, 1], result_seq=[0, 0, 0]),
      dict(causal=True, input_seq=[-1, 0, 1], result_seq=[-1, -0.5, 0]),
  ])
  def test_categorical_attn_can_implement_select_all(self, causal, input_seq,
                                                     result_seq):
    vocab = range(-20, 20)
    input_space = bases.VectorSpaceWithBasis.from_values("input", vocab)

    output_dir = bases.BasisDirection("output")
    output_space = bases.VectorSpaceWithBasis([output_dir])
    output_vec = output_space.vector_from_basis_direction(output_dir)

    bos_dir = bases.BasisDirection("bos_dimension")
    bos_space = bases.VectorSpaceWithBasis([bos_dir])

    one_dir = bases.BasisDirection("one")
    one_space = bases.VectorSpaceWithBasis([one_dir])

    value_dir = bases.BasisDirection("value")
    value_space = bases.VectorSpaceWithBasis([value_dir])

    input_space = bases.join_vector_spaces(input_space, bos_space, one_space)
    value_space = bases.join_vector_spaces(value_space, bos_space)
    residual_space = bases.join_vector_spaces(input_space, value_space,
                                              output_space)
    one_vec = residual_space.vector_from_basis_direction(one_dir)
    bos_vec = residual_space.vector_from_basis_direction(bos_dir)
    value_vec = residual_space.vector_from_basis_direction(value_dir)

    attn = categorical_attn.categorical_attn(
        key_space=input_space,
        query_space=input_space,
        value_space=value_space,
        output_space=output_space,
        bos_space=bos_space,
        one_space=one_space,
        attn_fn=lambda x, y: True,
        causal=causal)

    test_inputs = [bos_vec + one_vec]
    for x in input_seq:
      test_inputs.append(
          residual_space.vector_from_basis_direction(
              bases.BasisDirection("input", x)) + x * value_vec)
    test_inputs = bases.VectorInBasis.stack(test_inputs)

    # Expect the average of all (previous) tokens
    expected_results = [x * output_vec for x in result_seq]
    expected_results = bases.VectorInBasis.stack(expected_results)

    test_outputs = attn.apply(test_inputs).project(output_space)

    self.assertVectorAllClose(
        tests_common.strip_bos_token(test_outputs), expected_results)

  @parameterized.parameters([
      dict(causal=False, input_seq=[1, 2, 3, 4, 5], default=0),
      dict(causal=True, input_seq=[1, 2, 3, 4, 5], default=1),
      dict(causal=False, input_seq=[10], default=2),
      dict(causal=True, input_seq=[10], default=-3),
      dict(causal=False, input_seq=[-1, 0, 1], default=-2),
      dict(causal=True, input_seq=[-1, 0, 1], default=-1),
  ])
  def test_categorical_attn_can_implement_select_none(self, causal, input_seq,
                                                      default):
    vocab = range(-20, 20)
    input_space = bases.VectorSpaceWithBasis.from_values("input", vocab)

    output_dir = bases.BasisDirection("output")
    output_space = bases.VectorSpaceWithBasis([output_dir])
    default_vec = default * output_space.vector_from_basis_direction(output_dir)

    bos_dir = bases.BasisDirection("bos_dimension")
    bos_space = bases.VectorSpaceWithBasis([bos_dir])

    one_dir = bases.BasisDirection("one")
    one_space = bases.VectorSpaceWithBasis([one_dir])

    value_dir = bases.BasisDirection("value")
    value_space = bases.VectorSpaceWithBasis([value_dir])

    input_space = bases.join_vector_spaces(input_space, bos_space, one_space)
    value_space = bases.join_vector_spaces(value_space, bos_space)
    residual_space = bases.join_vector_spaces(input_space, value_space,
                                              output_space)
    value_vec = residual_space.vector_from_basis_direction(value_dir)
    bos_vec = residual_space.vector_from_basis_direction(bos_dir)
    one_vec = residual_space.vector_from_basis_direction(one_dir)

    attn = categorical_attn.categorical_attn(
        key_space=input_space,
        query_space=input_space,
        value_space=value_space,
        output_space=output_space,
        bos_space=bos_space,
        one_space=one_space,
        attn_fn=lambda x, y: False,
        default_output=default_vec,
        causal=causal,
        always_attend_to_bos=False,
        use_bos_for_default_output=True)

    def make_input(x):
      return (one_vec + x * value_vec +
              residual_space.vector_from_basis_direction(
                  bases.BasisDirection("input", x)))

    test_inputs = bases.VectorInBasis.stack([bos_vec + one_vec] +
                                            [make_input(x) for x in input_seq])

    # Expect the default value
    expected_results = [default_vec for x in input_seq]
    expected_results = bases.VectorInBasis.stack(expected_results)

    test_outputs = attn.apply(test_inputs).project(output_space)

    self.assertVectorAllClose(
        tests_common.strip_bos_token(test_outputs), expected_results)

  @parameterized.parameters([
      dict(num_counts=5, input_seq=[1, 4, 3, 2], n=1, result=[4, 3, 2, 1]),
      dict(num_counts=10, input_seq=[5, 8, 9, 2], n=3, result=[2, 5, 8, 9])
  ])
  def test_categorical_attn_can_implement_shift_by_n(self, num_counts,
                                                     input_seq, n, result):
    query_prefix = "prefix1"
    key_prefix = "prefix2"
    agg_input_prefix = "prefix3"
    output_prefix = "prefix4"

    bos_direction = bases.BasisDirection("bos")
    one_direction = bases.BasisDirection("one")
    query_space = bases.VectorSpaceWithBasis.from_values(
        query_prefix, range(num_counts))
    key_space = bases.VectorSpaceWithBasis.from_values(key_prefix,
                                                       range(num_counts))
    bos_space = bases.VectorSpaceWithBasis([bos_direction])
    one_space = bases.VectorSpaceWithBasis([one_direction])
    key_space = bases.join_vector_spaces(key_space, bos_space)

    agg_input_space = bases.VectorSpaceWithBasis.from_values(
        agg_input_prefix, range(num_counts))
    agg_input_space = bases.join_vector_spaces(agg_input_space, bos_space)
    output_space = bases.VectorSpaceWithBasis.from_values(
        output_prefix, range(num_counts))

    attn = categorical_attn.categorical_attn(
        query_space=query_space,
        key_space=key_space,
        value_space=agg_input_space,
        output_space=output_space,
        bos_space=bos_space,
        one_space=one_space,
        attn_fn=lambda q, k: q.value == k.value,
        default_output=None,
        always_attend_to_bos=False,
        use_bos_for_default_output=True,
        causal=False)

    residual_space = bases.join_vector_spaces(key_space, query_space,
                                              agg_input_space, output_space,
                                              one_space)

    seq_len = len(input_seq)
    query_seq = np.arange(n, seq_len + n) % seq_len
    key_seq = np.arange(seq_len)

    bos_vec = residual_space.vector_from_basis_direction(bos_direction)
    one_vec = residual_space.vector_from_basis_direction(one_direction)

    test_inputs = [bos_vec + one_vec]
    expected_results = []
    for i in range(seq_len):
      test_inputs.append(
          residual_space.vector_from_basis_direction(
              bases.BasisDirection(query_prefix, query_seq[i])) +
          residual_space.vector_from_basis_direction(
              bases.BasisDirection(key_prefix, key_seq[i])) +
          residual_space.vector_from_basis_direction(
              bases.BasisDirection(agg_input_prefix, input_seq[i])))
      expected_results.append(
          residual_space.vector_from_basis_direction(
              bases.BasisDirection(output_prefix, result[i])))

    test_inputs = bases.VectorInBasis.stack(test_inputs)
    expected_results = bases.VectorInBasis.stack(expected_results)

    test_outputs = attn.apply(test_inputs)

    self.assertVectorAllClose(
        tests_common.strip_bos_token(test_outputs), expected_results)


if __name__ == "__main__":
  absltest.main()
