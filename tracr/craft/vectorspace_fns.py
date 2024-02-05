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
"""Functions on vector spaces."""

import abc
import dataclasses
from typing import Callable, Sequence

import numpy as np
from tracr.craft import bases

VectorSpaceWithBasis = bases.VectorSpaceWithBasis
VectorInBasis = bases.VectorInBasis
BasisDirection = bases.BasisDirection


class VectorFunction(abc.ABC):
  """A function that acts on vectors."""

  input_space: VectorSpaceWithBasis
  output_space: VectorSpaceWithBasis

  @abc.abstractmethod
  def __call__(self, x: VectorInBasis) -> VectorInBasis:
    """Evaluates the function."""


class Linear(VectorFunction):
  """A linear function."""

  def __init__(
      self,
      input_space: VectorSpaceWithBasis,
      output_space: VectorSpaceWithBasis,
      matrix: np.ndarray,
  ):
    """Initialises.

    Args:
      input_space: The input vector space.
      output_space: The output vector space.
      matrix: a [input, output] matrix acting in a (sorted) basis.
    """
    self.input_space = input_space
    self.output_space = output_space
    self.matrix = matrix

  def __post_init__(self) -> None:
    output_size, input_size = self.matrix.shape
    assert input_size == self.input_space.num_dims
    assert output_size == self.output_space.num_dims

  def __call__(self, x: VectorInBasis) -> VectorInBasis:
    if x not in self.input_space:
      raise TypeError(f"x={x} not in self.input_space={self.input_space}.")
    return self.output_space.make_vector(x.magnitudes @ self.matrix)

  @classmethod
  def from_action(
      cls,
      input_space: VectorSpaceWithBasis,
      output_space: VectorSpaceWithBasis,
      action: Callable[[BasisDirection], VectorInBasis],
  ) -> "Linear":
    """from_action(i, o)(action) creates a Linear."""

    matrix = np.zeros((input_space.num_dims, output_space.num_dims))
    for i, direction in enumerate(input_space.basis):
      out_vector = action(direction)
      if out_vector not in output_space:
        raise TypeError(
            f"image of {direction} from input_space={input_space} "
            f"is not in output_space={output_space}"
        )
      matrix[i, :] = out_vector.magnitudes

    return Linear(input_space, output_space, matrix)

  @classmethod
  def combine_in_parallel(cls, fns: Sequence["Linear"]) -> "Linear":
    """Combines multiple parallel linear functions into a single one."""
    joint_input_space = bases.join_vector_spaces(
        *[fn.input_space for fn in fns]
    )
    joint_output_space = bases.join_vector_spaces(
        *[fn.output_space for fn in fns]
    )

    # Cache properties for the parents to avoid recomputing for each child.
    # Since the index_by_direction cached_property of the children is needed
    # within the action, it would be computed for every single child. This is
    # redundant as they share the same basis. By accessing the properties here,
    # we ensure they are only computed once and passed on to the children.
    _ = joint_input_space.index_by_direction
    _ = joint_output_space.index_by_direction

    def action(x: bases.BasisDirection) -> bases.VectorInBasis:
      out = joint_output_space.null_vector()
      for fn in fns:
        if x in fn.input_space:
          x_vec = fn.input_space.vector_from_basis_direction(x)
          applied = fn(x_vec)
          out = out.add_directions(applied)
      return out

    return cls.from_action(joint_input_space, joint_output_space, action)


def project(
    from_space: VectorSpaceWithBasis,
    to_space: VectorSpaceWithBasis,
) -> Linear:
  """Creates a projection."""

  def action(direction: bases.BasisDirection) -> VectorInBasis:
    if direction in to_space:
      return to_space.vector_from_basis_direction(direction)
    else:
      return to_space.null_vector()

  return Linear.from_action(from_space, to_space, action=action)


@dataclasses.dataclass
class ScalarBilinear:
  """A scalar-valued bilinear operator."""

  left_space: VectorSpaceWithBasis
  right_space: VectorSpaceWithBasis
  matrix: np.ndarray

  def __post_init__(self):
    """Ensure matrix acts in sorted bases and typecheck sizes."""
    left_size, right_size = self.matrix.shape
    assert left_size == self.left_space.num_dims
    assert right_size == self.right_space.num_dims

  def __call__(self, x: VectorInBasis, y: VectorInBasis) -> float:
    """Describes the action of the operator on vectors."""
    if x not in self.left_space:
      raise TypeError(f"x={x} not in self.left_space={self.left_space}.")
    if y not in self.right_space:
      raise TypeError(f"y={y} not in self.right_space={self.right_space}.")
    return (x.magnitudes.T @ self.matrix @ y.magnitudes).item()

  @classmethod
  def from_action(
      cls,
      left_space: VectorSpaceWithBasis,
      right_space: VectorSpaceWithBasis,
      action: Callable[[BasisDirection, BasisDirection], float],
  ) -> "ScalarBilinear":
    """from_action(l, r)(action) creates a ScalarBilinear."""

    matrix = np.zeros((left_space.num_dims, right_space.num_dims))
    for i, left_direction in enumerate(left_space.basis):
      for j, right_direction in enumerate(right_space.basis):
        matrix[i, j] = action(left_direction, right_direction)

    return ScalarBilinear(left_space, right_space, matrix)
