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
"""Pieces for making transformers."""

import abc
import dataclasses
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

from tracr.craft import bases
from tracr.craft import vectorspace_fns

project = vectorspace_fns.project


def _np_softmax(x, axis=-1):
  x_max = np.max(x, axis=axis, keepdims=True)
  return np.exp(x - x_max) / np.sum(np.exp(x - x_max), axis=axis, keepdims=True)


def _np_relu(x):
  return np.where(x > 0, x, 0)


def relu(x: bases.VectorInBasis) -> bases.VectorInBasis:
  return bases.VectorInBasis(x.basis_directions, _np_relu(x.magnitudes))


class Block(abc.ABC):
  """Transformer block, acting on a sequence of vector space elements.

  Attributes:
    residual_space: Vector space that contains all subspaces the Block interacts
      with. This can be either the full residual space of a model or a subspace.
  """
  residual_space: bases.VectorSpaceWithBasis

  @abc.abstractmethod
  def apply(self, x: bases.VectorInBasis) -> bases.VectorInBasis:
    """Applies self to an input."""


@dataclasses.dataclass
class AttentionHead(Block):
  """A transformer attention head."""
  w_qk: vectorspace_fns.ScalarBilinear
  w_ov: vectorspace_fns.Linear
  residual_space: Optional[bases.VectorSpaceWithBasis] = None
  causal: bool = False

  def __post_init__(self):
    """Infer residual stream and typecheck subspaces."""
    if self.residual_space is None:
      self.residual_space = bases.join_vector_spaces(self.w_qk.left_space,
                                                     self.w_qk.right_space,
                                                     self.w_ov.input_space,
                                                     self.w_ov.output_space)

    assert self.w_qk.left_space.issubspace(self.residual_space)
    assert self.w_qk.right_space.issubspace(self.residual_space)
    assert self.w_ov.input_space.issubspace(self.residual_space)
    assert self.w_ov.output_space.issubspace(self.residual_space)

  def apply(self, x: bases.VectorInBasis) -> bases.VectorInBasis:
    assert self.residual_space is not None
    assert x in self.residual_space
    # seq_len x query_space
    queries = x.project(self.w_qk.left_space)
    # seq_len x key_space
    keys = x.project(self.w_qk.right_space)

    attn_matrix = queries.magnitudes @ self.w_qk.matrix @ keys.magnitudes.T

    if self.causal:
      # The 1 gives us the matrix above the diagonal.
      mask = np.triu(np.full_like(attn_matrix, -np.inf), 1)
      attn_matrix = attn_matrix + mask

    attn_weights = _np_softmax(attn_matrix)  # seq_len_from, seq_len_to
    values = self.w_ov_residual(x).magnitudes  # seq_len_to, d_model

    magnitudes = attn_weights @ values  # seq_len_from, d_model
    return bases.VectorInBasis(sorted(self.residual_space.basis), magnitudes)

  def w_ov_residual(self, x: bases.VectorInBasis) -> bases.VectorInBasis:
    """Wov but acting on the residual space."""
    x = project(self.residual_space, self.w_ov.input_space)(x)
    out = self.w_ov(x)
    return project(self.w_ov.output_space, self.residual_space)(out)

  @property
  def num_heads(self) -> int:
    return 1

  def as_multi(self) -> "MultiAttentionHead":
    return MultiAttentionHead([self])


@dataclasses.dataclass
class MultiAttentionHead(Block):
  """Applies attention heads in parallel."""
  sub_blocks: List[Union[AttentionHead, "MultiAttentionHead"]]

  def __post_init__(self):
    spaces = [block.residual_space for block in self.sub_blocks]
    self.residual_space, *others = spaces
    assert all(s == self.residual_space for s in others)

  def apply(self, x: bases.VectorInBasis) -> bases.VectorInBasis:
    # each element is seq_len x embedding
    outs = [block.apply(x) for block in self.sub_blocks]
    return bases.VectorInBasis.sum(outs)  # seq_len x embedding

  @property
  def num_heads(self) -> int:
    return sum(sub_block.num_heads for sub_block in self.sub_blocks)

  def heads(self) -> Iterable[AttentionHead]:
    for sub_block in self.sub_blocks:
      if isinstance(sub_block, AttentionHead):
        yield sub_block
      elif isinstance(sub_block, MultiAttentionHead):
        yield from sub_block.heads()
      else:
        raise NotImplementedError()

  def as_multi(self) -> "MultiAttentionHead":
    return self


@dataclasses.dataclass
class MLP(Block):
  """A transformer MLP block."""
  fst: vectorspace_fns.Linear
  snd: vectorspace_fns.Linear
  residual_space: Optional[bases.VectorSpaceWithBasis] = None

  def __post_init__(self):
    """Typecheck subspaces."""
    if self.residual_space is None:
      self.residual_space = bases.join_vector_spaces(self.fst.input_space,
                                                     self.snd.output_space)

    assert self.fst.output_space == self.snd.input_space
    assert self.fst.input_space.issubspace(self.residual_space)
    assert self.snd.output_space.issubspace(self.residual_space)

  def apply(self, x: bases.VectorInBasis) -> bases.VectorInBasis:
    assert x in self.residual_space

    x = project(self.residual_space, self.fst.input_space)(x)
    hidden = self.fst(x)
    hidden = relu(hidden)
    out = self.snd(hidden)
    return project(self.snd.output_space, self.residual_space)(out)

  @classmethod
  def combine_in_parallel(cls, mlps: Sequence["MLP"]) -> "MLP":
    fst = vectorspace_fns.Linear.combine_in_parallel(
        [block.fst for block in mlps])
    snd = vectorspace_fns.Linear.combine_in_parallel(
        [block.snd for block in mlps])
    return cls(fst=fst, snd=snd, residual_space=None)


# Block that fits into a half-layer, without residual connections.
HalfLayerBlock = Union[MLP, AttentionHead, MultiAttentionHead]


@dataclasses.dataclass
class SeriesWithResiduals(Block):
  """A series of blocks with residual connections."""
  blocks: List[HalfLayerBlock]

  def __post_init__(self):
    spaces = [block.residual_space for block in self.blocks]
    self.residual_space = bases.join_vector_spaces(*spaces)

  def apply(self, x: bases.VectorInBasis) -> bases.VectorInBasis:
    x = x.project(self.residual_space)
    for block in self.blocks:
      x_in = x.project(block.residual_space)
      x_out = block.apply(x_in).project(self.residual_space)
      x = x + x_out
    return x
