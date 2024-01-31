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
"""Vectors and bases."""

import dataclasses
from typing import Sequence, Union, Optional, Iterable, Set, Dict
from functools import cached_property

import numpy as np

Name = Union[int, str]
Value = Union[int, float, bool, str, tuple]


@dataclasses.dataclass(frozen=True)
class BasisDirection:
  """Represents a basis direction (no magnitude) in a vector space.

  Attributes:
    name: a unique name for this direction.
    value: used to hold a value one-hot-encoded by this direction. e.g.,
      [BasisDirection("vs_1", True), BasisDirection("vs_1", False)] would be
      basis directions of a subspace called "vs_1" which one-hot-encodes the
      values True and False. If provided, considered part of the name for the
      purpose of disambiguating directions.
  """
  name: Name
  value: Optional[Value] = None

  def __str__(self):
    if self.value is None:
      return str(self.name)
    return f"{self.name}:{self.value}"

  def __lt__(self, other: "BasisDirection") -> bool:
    try:
      return (self.name, self.value) < (other.name, other.value)
    except TypeError:
      return str(self) < str(other)
  
  def __eq__(self, other):
    return (self.name, self.value) == (other.name, other.value)

@dataclasses.dataclass
class VectorInBasis:
  """A vector (or array of vectors) in a given basis.

  When magnitudes are 1-d, this is a vector.
  When magnitudes are (n+1)-d, this is an array of vectors,
  where the -1th dimension is the basis dimension.
  
  basis_is_sorted should remain False unless the basis is known to be ordered
  """
  basis_directions: Sequence[BasisDirection]
  magnitudes: np.ndarray
  _basis_is_sorted: bool = False
  def __post_init__(self):
    """Sort basis directions."""
    if len(self.basis_directions) != self.magnitudes.shape[-1]:
      raise ValueError(
          "Last dimension of magnitudes must be the same as number "
          f"of basis directions. Was {len(self.basis_directions)} "
          f"and {self.magnitudes.shape[-1]}.")

    if not self._basis_is_sorted:
      sort_idx = np.argsort(self.basis_directions)
      self.basis_directions = [self.basis_directions[i] for i in sort_idx]
      self.magnitudes = np.take(self.magnitudes, sort_idx, -1)
      self._basis_is_sorted = True
    
  @cached_property
  def basis_set(self) -> Set[BasisDirection]:
    """basis_directions stored in a set for faster lookups"""
    return set(self.basis_directions)
  
  @cached_property
  def direction_to_index(self) -> Dict[BasisDirection, int]:
    """Dictionary mapping Basis Directions to their index in self.basis_directions."""
    return {direction: index for index, direction in enumerate(self.basis_directions)}
  

  def __add__(self, other: "VectorInBasis") -> "VectorInBasis":
    if self.basis_directions != other.basis_directions:
      raise TypeError(f"Adding incompatible bases: {self} + {other}")
    magnitudes = self.magnitudes + other.magnitudes
    return self.update_magnitudes(magnitudes)

  def __radd__(self, other: "VectorInBasis") -> "VectorInBasis":
    if self.basis_directions != other.basis_directions:
      raise TypeError(f"Adding incompatible bases: {other} + {self}")
    return self + other

  def __sub__(self, other: "VectorInBasis") -> "VectorInBasis":
    if self.basis_directions != other.basis_directions:
      raise TypeError(f"Subtracting incompatible bases: {self} - {other}")
    magnitudes = self.magnitudes - other.magnitudes
    return self.update_magnitudes( magnitudes)

  def __rsub__(self, other: "VectorInBasis") -> "VectorInBasis":
    if self.basis_directions != other.basis_directions:
      raise TypeError(f"Subtracting incompatible bases: {other} - {self}")
    magnitudes = other.magnitudes - self.magnitudes
    return self.update_magnitudes(magnitudes)

  def __mul__(self, scalar: float) -> "VectorInBasis":
    return self.update_magnitudes(self.magnitudes * scalar)

  def __rmul__(self, scalar: float) -> "VectorInBasis":
    return self * scalar

  def __truediv__(self, scalar: float) -> "VectorInBasis":
    return self.update_magnitudes(self.magnitudes / scalar)

  def __neg__(self) -> "VectorInBasis":
    return (-1) * self

  def __eq__(self, other: "VectorInBasis") -> bool:
    return ((self.basis_directions == other.basis_directions) and
            (self.magnitudes.shape == other.magnitudes.shape) and
            (np.all(self.magnitudes == other.magnitudes)))

  @classmethod
  def sum(cls, vectors: Sequence["VectorInBasis"]) -> "VectorInBasis":
    return vectors[0].update_magnitudes(
               np.sum([x.magnitudes for x in vectors], axis=0))

  @classmethod
  def stack(cls,
            vectors: Sequence["VectorInBasis"],
            axis: int = 0) -> "VectorInBasis":
    for v in vectors[1:]:
      if v.basis_directions != vectors[0].basis_directions:
        raise TypeError(f"Stacking incompatible bases: {vectors[0]} + {v}")
    return vectors[0].update_magnitudes(
               np.stack([v.magnitudes for v in vectors], axis=axis))

  def project(
      self, basis_container: Union["VectorSpaceWithBasis", Sequence[BasisDirection]]
  ) -> "VectorInBasis":
    """Projects to the basis."""
    if isinstance(basis_container, VectorSpaceWithBasis):
      basis_directions = basis_container.basis
    else:
      basis_directions = basis_container
    components = []
    
    direction_to_index = self.direction_to_index
    for direction in basis_directions:
      # use a set for lookups, greatly improves performance
      if direction in self.basis_set:
        components.append(
            self.magnitudes[..., direction_to_index[direction]])
      else:
        components.append(np.zeros_like(self.magnitudes[..., 0]))
    
    # Check if basis is a VectorSpaceWithBasis or just a list of basis directions
    if isinstance(basis_container, VectorSpaceWithBasis):
      # if so, we can save compute and construct the vector in basis from the vector space
      return basis_container.make_vector(np.stack(components, axis=-1))
    else: 
      # otherwise construct one from scratch, we may have to build the set and index dict if requested
      return VectorInBasis(list(basis_container), np.stack(components, axis=-1), _basis_is_sorted=True)
  
  def add_directions(self, vector: "VectorInBasis") -> "VectorInBasis":
    """
    Equivalent to self += vector.project(self.basis).
    This is to speed up accumulation into an output vector in the combine_in_parallel in vectorspace_fns
    """
    directions_indices = self.direction_to_index
    new_magnitudes = np.array(self.magnitudes)
    for idx, basis in enumerate(vector.basis_directions):
      new_magnitudes[..., directions_indices[basis]] += vector.magnitudes[..., idx]
    return self.update_magnitudes(new_magnitudes)
  
  def update_magnitudes(self, new_magnitudes: np.ndarray) -> "VectorInBasis":
    """Create a new VectorInBasis object from this with update magnitudes.

    Will maintain any cached attributes we have computed prior
    """
    vect = VectorInBasis(list(self.basis_directions), new_magnitudes, _basis_is_sorted=True)
    if "basis_set" in vars(self):
      # we have spent the time to compute the basis set, so keep it in the copy
      vect.basis_set = self.basis_set
    if "direction_to_index" in vars(self):
      # we have spent the time to compute the direction_to_index dictionary, so keep it in the copy
      vect.direction_to_index = self.direction_to_index
    return vect
  
@dataclasses.dataclass
class VectorSpaceWithBasis:
  """A vector subspace in a given basis."""
  basis: Sequence[BasisDirection]

  def __post_init__(self):
    """Keep basis directions sorted."""
    self.basis = sorted(self.basis)
    
  def make_vector(self, magnitudes: np.ndarray) -> VectorInBasis:
    """Creates a VectorInBasis from our basis and provided magnitudes."""
    vect = VectorInBasis(list(self.basis), magnitudes=magnitudes, _basis_is_sorted=True)
    return vect
    
  @property
  def num_dims(self) -> int:
    return len(self.basis)
  
  @cached_property
  def basis_set(self):
    return set(self.basis)
  
  @cached_property
  def direction_to_index(self):
    return {direction: index for index, direction in enumerate(self.basis)}

  def __contains__(self, item: Union[VectorInBasis, BasisDirection]) -> bool:
    if isinstance(item, BasisDirection):
      return item in self.basis_set
    # item is a VectorInBasis, so it must also be sorted under the same ordering so we can just check the lists
    return self.basis == item.basis_directions

  def issubspace(self, other: "VectorSpaceWithBasis") -> bool:
    return self.basis_set.issubset(other.basis_set)

  def basis_vectors(self) -> Sequence[VectorInBasis]:
    basis_vector_magnitudes = list(np.eye(self.num_dims))
    return [self.make_vector(m) for m in basis_vector_magnitudes]

  def vector_from_basis_direction(
      self, basis_direction: BasisDirection) -> VectorInBasis:
    i = self.direction_to_index[basis_direction]
    return self.make_vector(np.eye(self.num_dims)[i])

  def null_vector(self) -> VectorInBasis:
    return self.make_vector(np.zeros(self.num_dims))

  @classmethod
  def from_names(cls, names: Sequence[Name]) -> "VectorSpaceWithBasis":
    """Creates a VectorSpace from a list of names for its basis directions."""
    return cls([BasisDirection(n) for n in names])

  @classmethod
  def from_values(
      cls,
      name: Name,
      values: Iterable[Value],
  ) -> "VectorSpaceWithBasis":
    """Creates a VectorSpace from a list of values for its basis directions."""
    return cls([BasisDirection(name, v) for v in values])


def direct_sum(*vs: VectorSpaceWithBasis) -> VectorSpaceWithBasis:
  """Create a direct sum of the vector spaces.

  Assumes the basis elements of all input vector spaces are
  orthogonal to each other. Maintains the order of the bases.

  Args:
    *vs: the vector spaces to sum.

  Returns:
    the combined vector space.

  Raises:
    Value error in case of overlapping bases.
  """
  # Take the union of all the bases:
  total_basis = sum([v.basis for v in vs], [])

  if len(total_basis) != len(set(total_basis)):
    raise ValueError("Overlapping bases!")

  return VectorSpaceWithBasis(total_basis)


def join_vector_spaces(*vs: VectorSpaceWithBasis) -> VectorSpaceWithBasis:
  """Joins a set of vector spaces allowing them to overlap.

  Assumes the basis elements of all input vector spaces are
  orthogonal to each other. Does not maintain the order of the bases but
  sorts them.

  Args:
    *vs: the vector spaces to sum.

  Returns:
    the combined vector space.
  """
  # Take the union of all the bases:
  total_basis = list(set().union(*[set(v.basis) for v in vs]))
  total_basis = sorted(total_basis)
  return VectorSpaceWithBasis(total_basis)


def ensure_dims(
    vs: VectorSpaceWithBasis,
    num_dims: int,
    name: str = "vector space",
) -> None:
  """Raises ValueError if vs has the wrong number of dimensions."""
  if vs.num_dims != num_dims:
    raise ValueError(f"{name} must have num_dims={num_dims}, "
                     f"but got {vs.num_dims}: {vs.basis}")
