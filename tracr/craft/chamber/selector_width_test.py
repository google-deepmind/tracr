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
"""Tests for selector_width."""

from absl.testing import absltest
from absl.testing import parameterized
from tracr.craft import bases
from tracr.craft import tests_common
from tracr.craft.chamber import selector_width


class SelectorWidthTest(tests_common.VectorFnTestCase):

  @parameterized.product(
      causal=[False, True],
      categorical_output=[False, True],
      input_seq=[[1, 2, 3, 4, 5], [-1, 0, 1], [10]],
  )
  def test_selector_width_of_select_all_is_length(
      self, causal, categorical_output, input_seq
  ):
    vocab = range(-20, 20)
    input_space = bases.VectorSpaceWithBasis.from_values("input", vocab)

    if categorical_output:
      output_space = bases.VectorSpaceWithBasis.from_values("output", range(10))
    else:
      output_space = bases.VectorSpaceWithBasis(
          [bases.BasisDirection("output")]
      )

    bos_dir = bases.BasisDirection("bos_dimension")
    bos_space = bases.VectorSpaceWithBasis([bos_dir])

    one_dir = bases.BasisDirection("one_dimension")
    one_space = bases.VectorSpaceWithBasis([one_dir])

    input_space = bases.join_vector_spaces(input_space, bos_space, one_space)
    residual_space = bases.join_vector_spaces(input_space, output_space)
    bos_vec = residual_space.vector_from_basis_direction(bos_dir)
    one_vec = residual_space.vector_from_basis_direction(one_dir)

    block = selector_width.selector_width(
        query_space=input_space,
        key_space=input_space,
        output_space=output_space,
        bos_space=bos_space,
        one_space=one_space,
        attn_fn=lambda x, y: True,
        out_value_set=set(range(len(input_seq) + 1)),
        categorical_output=categorical_output,
        causal=causal,
        label="select_all",
    )

    test_inputs = [bos_vec + one_vec]
    for x in input_seq:
      test_inputs.append(
          residual_space.vector_from_basis_direction(
              bases.BasisDirection("input", x)
          )
          + one_vec
      )
    test_inputs = bases.VectorInBasis.stack(test_inputs)

    # Expect length of the input sequence
    if causal:
      expected_results = list(range(1, len(input_seq) + 1))
    else:
      expected_results = [len(input_seq) for _ in input_seq]

    if categorical_output:
      expected_results = [
          output_space.vector_from_basis_direction(
              bases.BasisDirection("output", x)
          )
          for x in expected_results
      ]
    else:
      output_vec = output_space.vector_from_basis_direction(
          bases.BasisDirection("output")
      )
      expected_results = [x * output_vec for x in expected_results]

    expected_results = bases.VectorInBasis.stack(expected_results)

    test_outputs = block.apply(test_inputs).project(output_space)
    self.assertVectorAllClose(
        tests_common.strip_bos_token(test_outputs), expected_results
    )

  @parameterized.product(
      causal=[False, True],
      categorical_output=[False, True],
      input_seq=[[1] * 20, [2] * 50],
  )
  def test_selector_width_works_for_long_sequences(
      self, causal, categorical_output, input_seq
  ):
    vocab = range(-20, 20)
    input_space = bases.VectorSpaceWithBasis.from_values("input", vocab)

    if categorical_output:
      output_space = bases.VectorSpaceWithBasis.from_values(
          "output", range(100)
      )
    else:
      output_space = bases.VectorSpaceWithBasis(
          [bases.BasisDirection("output")]
      )

    bos_dir = bases.BasisDirection("bos_dimension")
    bos_space = bases.VectorSpaceWithBasis([bos_dir])

    one_dir = bases.BasisDirection("one_dimension")
    one_space = bases.VectorSpaceWithBasis([one_dir])

    input_space = bases.join_vector_spaces(input_space, bos_space, one_space)

    try:
      selector_width.selector_width(
          query_space=input_space,
          key_space=input_space,
          output_space=output_space,
          bos_space=bos_space,
          one_space=one_space,
          attn_fn=lambda x, y: True,
          out_value_set=set(range(len(input_seq) + 1)),
          categorical_output=categorical_output,
          causal=causal,
          label="select_all",
      )
    except AssertionError as e:
      if "output value mismatch" in str(e):
        # assertion raised if there are inconsistency in the expected output
        # values (likely due to floating point issues for long sequence length)
        self.fail(str(e))
      else:
        raise e

  @parameterized.product(
      causal=[False, True],
      categorical_output=[False, True],
      input_seq=[[1, 2, 3, 4, 5], [-1, 0, 1], [10]],
  )
  def test_selector_width_of_select_none_is_zero(
      self, causal, categorical_output, input_seq
  ):
    vocab = range(-20, 20)
    input_space = bases.VectorSpaceWithBasis.from_values("input", vocab)

    if categorical_output:
      output_space = bases.VectorSpaceWithBasis.from_values("output", range(10))
    else:
      output_space = bases.VectorSpaceWithBasis(
          [bases.BasisDirection("output")]
      )

    bos_dir = bases.BasisDirection("bos_dimension")
    bos_space = bases.VectorSpaceWithBasis([bos_dir])

    one_dir = bases.BasisDirection("one_dimension")
    one_space = bases.VectorSpaceWithBasis([one_dir])

    input_space = bases.join_vector_spaces(input_space, bos_space, one_space)
    residual_space = bases.join_vector_spaces(input_space, output_space)
    bos_vec = residual_space.vector_from_basis_direction(bos_dir)
    one_vec = residual_space.vector_from_basis_direction(one_dir)

    block = selector_width.selector_width(
        query_space=input_space,
        key_space=input_space,
        output_space=output_space,
        bos_space=bos_space,
        one_space=one_space,
        attn_fn=lambda x, y: False,
        out_value_set=set(range(len(input_seq) + 1)),
        categorical_output=categorical_output,
        causal=causal,
        label="select_all",
    )

    test_inputs = [bos_vec + one_vec]
    for x in input_seq:
      test_inputs.append(
          residual_space.vector_from_basis_direction(
              bases.BasisDirection("input", x)
          )
          + one_vec
      )
    test_inputs = bases.VectorInBasis.stack(test_inputs)

    # Expect zero output
    if categorical_output:
      expected_results = [
          output_space.vector_from_basis_direction(
              bases.BasisDirection("output", 0)
          )
          for _ in input_seq
      ]
    else:
      expected_results = [output_space.null_vector() for _ in input_seq]
    expected_results = bases.VectorInBasis.stack(expected_results)

    test_outputs = block.apply(test_inputs).project(output_space)
    self.assertVectorAllClose(
        tests_common.strip_bos_token(test_outputs), expected_results
    )


if __name__ == "__main__":
  absltest.main()
