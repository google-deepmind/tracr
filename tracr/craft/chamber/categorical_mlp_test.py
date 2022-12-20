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
"""Tests for chamber.categorical_mlp."""

import math
from absl.testing import absltest
from absl.testing import parameterized

from tracr.craft import bases
from tracr.craft import tests_common
from tracr.craft.chamber import categorical_mlp


class CategoricalInputMlpTest(tests_common.VectorFnTestCase):

  @parameterized.parameters([
      dict(num_counts=4, x=1, y=2, fun=lambda x, y: x + y, result=3),
      dict(num_counts=4, x=1, y=0, fun=lambda x, y: x + y + 1, result=2),
      dict(num_counts=5, x=2, y=1, fun=math.pow, result=2),
      dict(num_counts=5, x=2, y=2, fun=math.pow, result=4),
  ])
  def test_seq_map_categorical_mlp_produces_expected_outcome(
      self, num_counts, x, y, fun, result):
    input1_name = "in1"
    input2_name = "in2"
    output_name = "out"
    one_name = "one_dimension"

    in1_space = bases.VectorSpaceWithBasis.from_values(input1_name,
                                                       range(num_counts + 1))
    in2_space = bases.VectorSpaceWithBasis.from_values(input2_name,
                                                       range(num_counts + 1))
    out_space = bases.VectorSpaceWithBasis.from_values(output_name,
                                                       range(num_counts + 1))

    def operation(in1, in2):
      out_val = fun(int(in1.value), int(in2.value))
      return bases.BasisDirection(output_name, out_val)

    mlp = categorical_mlp.sequence_map_categorical_mlp(
        input1_space=in1_space,
        input2_space=in2_space,
        output_space=out_space,
        operation=operation,
        one_space=bases.VectorSpaceWithBasis.from_names([one_name]))

    test_inputs = (
        mlp.residual_space.vector_from_basis_direction(
            bases.BasisDirection(one_name)) +
        mlp.residual_space.vector_from_basis_direction(
            bases.BasisDirection(input1_name, x)) +
        mlp.residual_space.vector_from_basis_direction(
            bases.BasisDirection(input2_name, y)))

    expected_results = mlp.residual_space.vector_from_basis_direction(
        bases.BasisDirection(output_name, result))

    test_outputs = mlp.apply(test_inputs)

    self.assertVectorAllClose(test_outputs, expected_results)

  def test_seq_map_categorical_mlp_raises_error_with_overlapping_inputs(self):
    input_name = "in"
    output_name = "out"
    one_name = "one_dimension"

    in1_space = bases.VectorSpaceWithBasis.from_values(input_name, range(5))
    in2_space = bases.VectorSpaceWithBasis.from_values(input_name, range(3, 10))
    out_space = bases.VectorSpaceWithBasis.from_values(output_name, range(5))

    with self.assertRaisesRegex(
        ValueError, r".*Input spaces to a SequenceMap must be disjoint.*"):
      categorical_mlp.sequence_map_categorical_mlp(
          input1_space=in1_space,
          input2_space=in1_space,
          output_space=out_space,
          operation=lambda x, y: bases.BasisDirection(output_name, 0),
          one_space=bases.VectorSpaceWithBasis.from_names([one_name]))

    with self.assertRaisesRegex(
        ValueError, r".*Input spaces to a SequenceMap must be disjoint.*"):
      categorical_mlp.sequence_map_categorical_mlp(
          input1_space=in1_space,
          input2_space=in2_space,
          output_space=out_space,
          operation=lambda x, y: bases.BasisDirection(output_name, 0),
          one_space=bases.VectorSpaceWithBasis.from_names([one_name]))

  @parameterized.parameters([
      dict(num_counts=5, x=2, fun=lambda x: x, result=2),
      dict(num_counts=5, x=2, fun=lambda x: math.pow(x, int(2)), result=4),
      dict(num_counts=5, x=-2, fun=lambda x: math.pow(x, int(2)), result=4),
      dict(num_counts=5, x=-1, fun=lambda x: math.pow(x, int(3)), result=-1),
  ])
  def test_map_categorical_mlp_produces_expected_outcome_computing_powers(
      self, num_counts, x, fun, result):
    input_name = "in"
    output_name = "out"

    in_space = bases.VectorSpaceWithBasis.from_values(
        input_name, range(-num_counts, num_counts + 1))
    out_space = bases.VectorSpaceWithBasis.from_values(
        output_name, range(-num_counts, num_counts + 1))

    def operation(direction):
      out_val = fun(int(direction.value))
      return bases.BasisDirection(output_name, out_val)

    mlp = categorical_mlp.map_categorical_mlp(
        input_space=in_space, output_space=out_space, operation=operation)

    test_inputs = mlp.residual_space.vector_from_basis_direction(
        bases.BasisDirection(input_name, x))

    expected_results = mlp.residual_space.vector_from_basis_direction(
        bases.BasisDirection(output_name, result))

    test_outputs = mlp.apply(test_inputs)

    self.assertVectorAllClose(test_outputs, expected_results)

  @parameterized.parameters([
      dict(x=2, fun=lambda x: x, result=2),
      dict(x=2, fun=lambda x: math.pow(x, int(2)), result=4),
      dict(x=1, fun=lambda x: 1 / (x + 1), result=0.5),
      dict(x=3, fun=lambda x: 1 / (x + 1), result=0.25),
  ])
  def test_map_categorical_to_numerical_mlp_produces_expected_outcome(
      self, x, fun, result):

    in_space = bases.VectorSpaceWithBasis.from_values("in", range(6))
    out_space = bases.VectorSpaceWithBasis.from_names(["out"])

    mlp = categorical_mlp.map_categorical_to_numerical_mlp(
        input_space=in_space,
        output_space=out_space,
        operation=fun,
    )

    test_inputs = mlp.residual_space.vector_from_basis_direction(
        bases.BasisDirection("in", x))

    expected_results = result * mlp.residual_space.vector_from_basis_direction(
        bases.BasisDirection("out"))

    test_outputs = mlp.apply(test_inputs)

    self.assertVectorAllClose(test_outputs, expected_results)


if __name__ == "__main__":
  absltest.main()
