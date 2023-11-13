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
"""Inferring the vector spaces taken on by certain operations."""

import dataclasses
import itertools
from typing import Set

import networkx as nx
from tracr.compiler import nodes
from tracr.craft import bases
from tracr.rasp import rasp
from tracr.utils import errors

Node = nodes.Node


@dataclasses.dataclass
class InferBasesOutput:
  graph: nx.DiGraph


def infer_bases(
    graph: nx.DiGraph,
    sink: Node,
    vocab: Set[rasp.Value],
    max_seq_len: int,
) -> None:
  """Infers in-place the possible output values and vector bases of the SOps."""

  def compute_value_set(sop: rasp.SOp) -> Set[rasp.Value]:
    """Computes value set using already-computed predecessor value sets."""
    if isinstance(sop, rasp.TokensType):
      return vocab
    elif isinstance(sop, rasp.IndicesType):
      return set(range(max_seq_len))
    elif isinstance(sop, rasp.SelectorWidth):
      return set(range(0, max_seq_len + 1))
    elif isinstance(sop, rasp.Full):
      return {sop.fill}
    elif isinstance(sop, rasp.Map):
      inner_value_set = graph.nodes[sop.inner.label][nodes.VALUE_SET]
      out = set()
      for x in inner_value_set:
        res = errors.ignoring_arithmetic_errors(sop.f)(x)
        if res is not None:
          out.add(res)
      return out
    elif isinstance(sop, rasp.SequenceMap):
      f_ignore_error = errors.ignoring_arithmetic_errors(sop.f)
      fst_value_set = graph.nodes[sop.fst.label][nodes.VALUE_SET]
      snd_value_set = graph.nodes[sop.snd.label][nodes.VALUE_SET]
      out = set()
      for l, r in itertools.product(fst_value_set, snd_value_set):
        res = f_ignore_error(l, r)
        if res is not None:
          out.add(res)
      return out
    elif isinstance(sop, rasp.Aggregate):
      if rasp.is_categorical(sop):
        # Simply pass on the value set of the underlying S-Op.
        return graph.nodes[sop.sop.label][nodes.VALUE_SET]
      elif rasp.is_numerical(sop):
        # TODO(b/255936408): This doesn't work if we average arbitrary values.
        # But most examples only average binary variables.
        sop_value_set = graph.nodes[sop.sop.label][nodes.VALUE_SET]
        if not {int(x) for x in sop_value_set}.issubset({0, 1}):
          raise NotImplementedError(
              "Attention patterns can currently only "
              "average binary variables. Not:", sop_value_set)

        value_set = set()
        for value in sop_value_set:
          for length in range(1, max_seq_len + 1):
            value_set.add(value / length)
        return value_set
    raise ValueError(f"Unsupported S-Op: {sop}")

  for node_id in nx.dfs_postorder_nodes(graph.reverse(), sink[nodes.ID]):
    expr = graph.nodes[node_id][nodes.EXPR]

    if not isinstance(expr, rasp.SOp):
      # Only S-Ops have output vector spaces.
      continue

    value_set = compute_value_set(expr)
    graph.nodes[node_id][nodes.VALUE_SET] = value_set

    if rasp.is_categorical(expr):
      out_space = bases.VectorSpaceWithBasis.from_values(expr.label, value_set)
    elif rasp.is_numerical(expr):
      out_space = bases.VectorSpaceWithBasis.from_names([expr.label])
    else:
      raise ValueError(f"Unsupported S-Op type: {expr.type}")
    graph.nodes[node_id][nodes.OUTPUT_BASIS] = out_space.basis
