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
"""MLPs to compute arbitrary numerical functions by discretising."""

import dataclasses
from typing import Callable, Iterable, List

from tracr.craft import bases
from tracr.craft import transformers
from tracr.craft import vectorspace_fns
from tracr.utils import errors


@dataclasses.dataclass
class DiscretisingLayerMaterials:
  """Provides components for a hidden layer that discretises the input.

  Attributes:
    action: Function acting on basis directions that defines the computation.
    hidden_space: Vector space of the hidden representation of the layer.
    output_values: Set of output values that correspond to the discretisation.
  """

  action: Callable[[bases.BasisDirection], bases.VectorInBasis]
  hidden_space: bases.VectorSpaceWithBasis
  output_values: List[float]


def _get_discretising_layer(
    input_value_set: Iterable[float],
    f: Callable[[float], float],
    hidden_name: bases.Name,
    one_direction: bases.BasisDirection,
    large_number: float,
) -> DiscretisingLayerMaterials:
  """Creates a hidden layer that discretises the input of f(x) into a value set.

  The input is split up into a distinct region around each value in
  `input_value_set`:

  elements of value set:  v0   |  v1  |  v2  |  v3  |  v4  | ...
  thresholds:                  t0     t1     t2     t3     t4

  The hidden layer has two activations per threshold:
    hidden_k_1 = ReLU(L * (x - threshold[k]) + 1)
    hidden_k_2 = ReLU(L * (x - threshold[k]))

  Note that hidden_k_1 - hidden_k_2 is:
    1                 if x >= threshold[k] + 1/L
    0                 if x <= threshold[k]
    between 0 and 1   if threshold[k] < x < threshold[k] + 1/L

  So as long as we choose L a big enough number, we have
    hidden_k_1 - hidden_k_2 = 1 if x >= threshold[k].
  i.e. we know in which region the input value is.

  Args:
    input_value_set: Set of discrete input values.
    f: Function to approximate.
    hidden_name: Name for hidden dimensions.
    one_direction: Auxiliary dimension that must contain 1 in the input.
    large_number: Large number L that determines accuracy of the computation.

  Returns:
    DiscretisingLayerMaterials containing all components for the layer.
  """
  output_values, sorted_values = [], []
  for x in sorted(input_value_set):
    res = errors.ignoring_arithmetic_errors(f)(x)
    if res is not None:
      output_values.append(res)
      sorted_values.append(x)

  num_vals = len(sorted_values)
  value_thresholds = [
      (sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(num_vals - 1)
  ]

  hidden_directions = [bases.BasisDirection(f"{hidden_name}start")]
  for k in range(1, num_vals):
    dir0 = bases.BasisDirection(hidden_name, (k, 0))
    dir1 = bases.BasisDirection(hidden_name, (k, 1))
    hidden_directions.extend([dir0, dir1])
  hidden_space = bases.VectorSpaceWithBasis(hidden_directions)

  def action(direction: bases.BasisDirection) -> bases.VectorInBasis:
    # hidden_k_0 = ReLU(L * (x - threshold[k]) + 1)
    # hidden_k_1 = ReLU(L * (x - threshold[k]))
    if direction == one_direction:
      hidden = hidden_space.vector_from_basis_direction(
          bases.BasisDirection(f"{hidden_name}start")
      )
    else:
      hidden = hidden_space.null_vector()
    for k in range(1, num_vals):
      vec0 = hidden_space.vector_from_basis_direction(
          bases.BasisDirection(hidden_name, (k, 0))
      )
      vec1 = hidden_space.vector_from_basis_direction(
          bases.BasisDirection(hidden_name, (k, 1))
      )
      if direction == one_direction:
        hidden += (1 - large_number * value_thresholds[k - 1]) * vec0
        hidden -= large_number * value_thresholds[k - 1] * vec1
      else:
        hidden += large_number * vec0 + large_number * vec1
    return hidden

  return DiscretisingLayerMaterials(
      action=action, hidden_space=hidden_space, output_values=output_values
  )


def map_numerical_mlp(
    f: Callable[[float], float],
    input_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis,
    input_value_set: Iterable[float],
    one_space: bases.VectorSpaceWithBasis,
    large_number: float = 100,
    hidden_name: bases.Name = "__hidden__",
) -> transformers.MLP:
  """Returns an MLP that encodes any function of a single variable f(x).

  This is implemented by discretising the input according to input_value_set
  and defining thresholds that determine which part of the input range will
  is allocated to which value in input_value_set.

  elements of value set:  v0   |  v1  |  v2  |  v3  |  v4  | ...
  thresholds:                  t0     t1     t2     t3     t4

  The MLP computes two hidden activations per threshold:
    hidden_k_0 = ReLU(L * (x - threshold[k]) + 1)
    hidden_k_1 = ReLU(L * (x - threshold[k]))

  Note that hidden_k_1 - hidden_k_2 is:
    1                 if x >= threshold[k] + 1/L
    0                 if x <= threshold[k]
    between 0 and 1   if threshold[k] < x < threshold[k] + 1/L

  So as long as we choose L a big enough number, we have
    hidden_k_0 - hidden_k_1 = 1 if x >= threshold[k].

  The MLP then computes the output as:
    output = f(input[0]) +
      sum((hidden_k_0 - hidden_k_1) * (f(input[k]) - f(input[k-1]))
        for all k=0,1,...)

  This sum will be (by a telescoping sums argument)
    f(input[0])      if x <= threshold[0]
    f(input[k])      if threshold[k-1] < x <= threshold[k] for some other k
    f(input[-1])     if x > threshold[-1]
  which approximates f() up to an accuracy given by input_value_set and L.

  Args:
    f: Function to approximate.
    input_space: 1-d vector space that encodes the input x.
    output_space: 1-d vector space to write the output to.
    input_value_set: Set of values the input can take.
    one_space: Auxiliary 1-d vector space that must contain 1 in the input.
    large_number: Large number L that determines accuracy of the computation.
      Note that too large values of L can lead to numerical issues, particularly
      during inference on GPU/TPU.
    hidden_name: Name for hidden dimensions.
  """
  bases.ensure_dims(input_space, num_dims=1, name="input_space")
  bases.ensure_dims(output_space, num_dims=1, name="output_space")
  bases.ensure_dims(one_space, num_dims=1, name="one_space")

  input_space = bases.join_vector_spaces(input_space, one_space)
  out_vec = output_space.vector_from_basis_direction(output_space.basis[0])

  discretising_layer = _get_discretising_layer(
      input_value_set=input_value_set,
      f=f,
      hidden_name=hidden_name,
      one_direction=one_space.basis[0],
      large_number=large_number,
  )
  first_layer = vectorspace_fns.Linear.from_action(
      input_space, discretising_layer.hidden_space, discretising_layer.action
  )

  def second_layer_action(
      direction: bases.BasisDirection,
  ) -> bases.VectorInBasis:
    # output = sum(
    #     (hidden_k_0 - hidden_k_1) * (f(input[k]) - f(input[k-1]))
    #   for all k)
    if direction.name == f"{hidden_name}start":
      return discretising_layer.output_values[0] * out_vec
    k, i = direction.value
    # add hidden_k_0 and subtract hidden_k_1
    sign = {0: 1, 1: -1}[i]
    return (
        sign
        * (
            discretising_layer.output_values[k]
            - discretising_layer.output_values[k - 1]
        )
        * out_vec
    )

  second_layer = vectorspace_fns.Linear.from_action(
      discretising_layer.hidden_space, output_space, second_layer_action
  )

  return transformers.MLP(first_layer, second_layer)


def map_numerical_to_categorical_mlp(
    f: Callable[[float], float],
    input_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis,
    input_value_set: Iterable[float],
    one_space: bases.VectorSpaceWithBasis,
    large_number: float = 100,
    hidden_name: bases.Name = "__hidden__",
) -> transformers.MLP:
  """Returns an MLP to compute f(x) from a numerical to a categorical variable.

  Uses a set of possible output values, and rounds f(x) to the closest value
  in this set to create a categorical output variable.

  The output is discretised the same way as in `map_numerical_mlp`.

  Args:
    f: Function to approximate.
    input_space: 1-d vector space that encodes the input x.
    output_space: n-d vector space to write categorical output to. The output
      directions need to encode the possible output values.
    input_value_set: Set of values the input can take.
    one_space: Auxiliary 1-d space that must contain 1 in the input.
    large_number: Large number L that determines accuracy of the computation.
    hidden_name: Name for hidden dimensions.
  """
  bases.ensure_dims(input_space, num_dims=1, name="input_space")
  bases.ensure_dims(one_space, num_dims=1, name="one_space")

  input_space = bases.join_vector_spaces(input_space, one_space)

  vec_by_out_val = dict()
  for d in output_space.basis:
    # TODO(b/255937603): Do a similar assert in other places where we expect
    # categorical basis directions to encode values.
    assert (
        d.value is not None
    ), "output directions need to encode possible output values"
    vec_by_out_val[d.value] = output_space.vector_from_basis_direction(d)

  discretising_layer = _get_discretising_layer(
      input_value_set=input_value_set,
      f=f,
      hidden_name=hidden_name,
      one_direction=one_space.basis[0],
      large_number=large_number,
  )

  assert set(discretising_layer.output_values).issubset(
      set(vec_by_out_val.keys())
  ), (
      "output value mismatch. output_values:"
      f" {set(discretising_layer.output_values)}, vec_by_out_val:"
      f" {set(vec_by_out_val.keys())}"
  )

  first_layer = vectorspace_fns.Linear.from_action(
      input_space, discretising_layer.hidden_space, discretising_layer.action
  )

  def second_layer_action(
      direction: bases.BasisDirection,
  ) -> bases.VectorInBasis:
    """Computes output value and returns corresponding output direction."""
    if direction.name == f"{hidden_name}start":
      return vec_by_out_val[discretising_layer.output_values[0]]
    else:
      k, i = direction.value
      # add hidden_k_0 and subtract hidden_k_1
      sign = {0: 1, 1: -1}[i]
      out_k = discretising_layer.output_values[k]
      out_k_m_1 = discretising_layer.output_values[k - 1]
      return sign * (vec_by_out_val[out_k] - vec_by_out_val[out_k_m_1])

  second_layer = vectorspace_fns.Linear.from_action(
      discretising_layer.hidden_space, output_space, second_layer_action
  )

  return transformers.MLP(first_layer, second_layer)


def linear_sequence_map_numerical_mlp(
    input1_basis_direction: bases.BasisDirection,
    input2_basis_direction: bases.BasisDirection,
    output_basis_direction: bases.BasisDirection,
    input1_factor: float,
    input2_factor: float,
    hidden_name: bases.Name = "__hidden__",
) -> transformers.MLP:
  """Returns an MLP that encodes a linear function f(x, y) = a*x + b*y.

  Args:
    input1_basis_direction: Basis direction that encodes the input x.
    input2_basis_direction: Basis direction that encodes the input y.
    output_basis_direction: Basis direction to write the output to.
    input1_factor: Linear factor a for input x.
    input2_factor: Linear factor a for input y.
    hidden_name: Name for hidden dimensions.
  """
  input_space = bases.VectorSpaceWithBasis(
      [input1_basis_direction, input2_basis_direction]
  )
  output_space = bases.VectorSpaceWithBasis([output_basis_direction])
  out_vec = output_space.vector_from_basis_direction(output_basis_direction)

  hidden_directions = [
      bases.BasisDirection(f"{hidden_name}x", 1),
      bases.BasisDirection(f"{hidden_name}x", -1),
      bases.BasisDirection(f"{hidden_name}y", 1),
      bases.BasisDirection(f"{hidden_name}y", -1),
  ]
  hidden_space = bases.VectorSpaceWithBasis(hidden_directions)
  x_pos_vec, x_neg_vec, y_pos_vec, y_neg_vec = (
      hidden_space.vector_from_basis_direction(d) for d in hidden_directions
  )

  def first_layer_action(
      direction: bases.BasisDirection,
  ) -> bases.VectorInBasis:
    output = hidden_space.null_vector()
    if direction == input1_basis_direction:
      output += x_pos_vec - x_neg_vec
    if direction == input2_basis_direction:
      output += y_pos_vec - y_neg_vec
    return output

  first_layer = vectorspace_fns.Linear.from_action(
      input_space, hidden_space, first_layer_action
  )

  def second_layer_action(
      direction: bases.BasisDirection,
  ) -> bases.VectorInBasis:
    if direction.name == f"{hidden_name}x":
      return input1_factor * direction.value * out_vec
    if direction.name == f"{hidden_name}y":
      return input2_factor * direction.value * out_vec
    return output_space.null_vector()

  second_layer = vectorspace_fns.Linear.from_action(
      hidden_space, output_space, second_layer_action
  )

  return transformers.MLP(first_layer, second_layer)
