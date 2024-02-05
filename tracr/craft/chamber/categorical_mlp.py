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
"""MLP to compute basic linear functions of one-hot encoded integers."""

from typing import Callable

import numpy as np

from tracr.craft import bases
from tracr.craft import transformers
from tracr.craft import vectorspace_fns

_ONE_SPACE = bases.VectorSpaceWithBasis.from_names(["one"])


def map_categorical_mlp(
    input_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis,
    operation: Callable[[bases.BasisDirection], bases.BasisDirection],
) -> transformers.MLP:
  """Returns an MLP that encodes any categorical function of a single variable f(x).

  The hidden layer is the identity and output combines this with a lookup table
    output_k = sum(f(i)*input_i for all i in input space)

  Args:
    input_space: space containing the input x.
    output_space: space containing possible outputs.
    operation: A function operating on basis directions.
  """

  def operation_fn(direction):
    if direction in input_space:
      output_direction = operation(direction)
      if output_direction in output_space:
        return output_space.vector_from_basis_direction(output_direction)
    return output_space.null_vector()

  first_layer = vectorspace_fns.Linear.from_action(input_space, output_space,
                                                   operation_fn)

  second_layer = vectorspace_fns.project(output_space, output_space)

  return transformers.MLP(first_layer, second_layer)


def map_categorical_to_numerical_mlp(
    input_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis,
    operation: Callable[[bases.Value], float],
) -> transformers.MLP:
  """Returns an MLP to compute f(x) from a categorical to a numerical variable.

  The hidden layer is the identity and output combines this with a lookup table
    output = sum(f(i)*input_i for all i in input space)

  Args:
    input_space: Vector space containing the input x.
    output_space: Vector space to write the numerical output to.
    operation: A function operating on basis directions.
  """
  bases.ensure_dims(output_space, num_dims=1, name="output_space")
  out_vec = output_space.vector_from_basis_direction(output_space.basis[0])

  def operation_fn(direction):
    if direction in input_space:
      return operation(direction.value) * out_vec
    return output_space.null_vector()

  first_layer = vectorspace_fns.Linear.from_action(input_space, output_space,
                                                   operation_fn)

  second_layer = vectorspace_fns.project(output_space, output_space)

  return transformers.MLP(first_layer, second_layer)


def sequence_map_categorical_mlp(
    input1_space: bases.VectorSpaceWithBasis,
    input2_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis,
    operation: Callable[[bases.BasisDirection, bases.BasisDirection],
                        bases.BasisDirection],
    one_space: bases.VectorSpaceWithBasis = _ONE_SPACE,
    hidden_name: bases.Name = "__hidden__",
) -> transformers.MLP:
  """Returns an MLP that encodes a categorical function of two variables f(x, y).

  The hidden layer of the MLP computes the logical and of all input directions
    hidden_i_j = ReLU(x_i+x_j-1)

  And the output combines this with a lookup table
    output_k = sum(f(i, j)*hidden_i_j for all i,j in input space)

  Args:
    input1_space: Vector space containing the input x.
    input2_space: Vector space containing the input y.
    output_space: Vector space to write outputs to.
    operation: A function operating on basis directions.
    one_space: a reserved 1-d space that always contains a 1.
    hidden_name: Name for hidden dimensions.
  """
  bases.ensure_dims(one_space, num_dims=1, name="one_space")

  if not set(input1_space.basis).isdisjoint(input2_space.basis):
    raise ValueError("Input spaces to a SequenceMap must be disjoint. "
                     "If input spaces are the same, use Map instead!")

  input_space = bases.direct_sum(input1_space, input2_space, one_space)

  def to_hidden(x, y):
    return bases.BasisDirection(hidden_name, (x.name, x.value, y.name, y.value))

  def from_hidden(h):
    x_name, x_value, y_name, y_value = h.value
    x_dir = bases.BasisDirection(x_name, x_value)
    y_dir = bases.BasisDirection(y_name, y_value)
    return x_dir, y_dir

  hidden_dir = []
  for dir1 in input1_space.basis:
    for dir2 in input2_space.basis:
      hidden_dir.append(to_hidden(dir1, dir2))
  hidden_space = bases.VectorSpaceWithBasis(hidden_dir)

  def logical_and(direction):
    if direction in one_space:
      out = bases.VectorInBasis(hidden_space.basis,
                                -np.ones(hidden_space.num_dims))
    elif direction in input1_space:
      dir1 = direction
      out = hidden_space.null_vector()
      for dir2 in input2_space.basis:
        vector = bases.VectorInBasis(
            [to_hidden(dir1, dir2)], np.array([1]), _basis_is_sorted=True
        )
        out = out.add_directions(vector)
    else:
      dir2 = direction
      out = hidden_space.null_vector()
      for dir1 in input1_space.basis:
        vector = bases.VectorInBasis(
            [to_hidden(dir1, dir2)], np.array([1]), _basis_is_sorted=True
        )
        out = out.add_directions(vector)
    return out

  first_layer = vectorspace_fns.Linear.from_action(input_space, hidden_space,
                                                   logical_and)

  def operation_fn(direction):
    dir1, dir2 = from_hidden(direction)
    output_direction = operation(dir1, dir2)
    if output_direction in output_space:
      return output_space.vector_from_basis_direction(output_direction)
    else:
      return output_space.null_vector()

  second_layer = vectorspace_fns.Linear.from_action(hidden_space, output_space,
                                                    operation_fn)

  return transformers.MLP(first_layer, second_layer)
