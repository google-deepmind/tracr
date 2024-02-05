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
"""Create a craft model from a computational graph."""

import collections
from typing import Dict, List, Sequence

import networkx as nx
from tracr.compiler import nodes
from tracr.craft import bases
from tracr.craft import transformers
from tracr.rasp import rasp

Node = nodes.Node
NodeID = nodes.NodeID


def compute_computational_depth(
    graph: nx.DiGraph, sources_ids: Sequence[int]
) -> Dict[int, int]:
  """Returns the computational depth of each node in the graph.

  Given source nodes, runs DFS to tracr the maximum computational depth from all
    source nodes to every node in the graph.

  Non-SOp nodes do not count in the depth calculation.

  Disconnected nodes have depth -1.

  Args:
    graph: RASP computational graph where all nodes are annotated with # EXPR
      attributes set to rasp primitives
    sources_ids: Sequence of integers to measure computational depth against

  Returns:
    a dictionary mapping all graph nodes to a computational depth
  """
  computational_depth = {}

  def dfs(node_id, depth):
    if node_id in computational_depth:
      computational_depth[node_id] = max(depth, computational_depth[node_id])
    else:
      computational_depth[node_id] = depth

    for successor_id in graph.successors(node_id):
      if not isinstance(graph.nodes[successor_id][nodes.EXPR], rasp.SOp):
        dfs(successor_id, depth)
      else:
        dfs(successor_id, depth + 1)

  for source_id in sources_ids:
    dfs(source_id, depth=0)

  # ensure any disconnected nodes are given a depth -1
  disconnected_nodes = set(graph.nodes) - set(computational_depth.keys())
  for disconnected_node in disconnected_nodes:
    computational_depth[disconnected_node] = -1

  return computational_depth


def _node_is_attn(node: Node) -> bool:
  """Returns True if node is an attention layer."""
  return nodes.MODEL_BLOCK in node and isinstance(
      node[nodes.MODEL_BLOCK],
      (transformers.AttentionHead, transformers.MultiAttentionHead),
  )


def _node_is_mlp(node: Node) -> bool:
  """Returns True if node is an MLP layer."""
  return nodes.MODEL_BLOCK in node and isinstance(
      node[nodes.MODEL_BLOCK], transformers.MLP
  )


def _node_is_residual_block(node: Node) -> bool:
  """Returns True if node is a valid residual block (Attn followed by MLP)."""
  block = node[nodes.MODEL_BLOCK] if nodes.MODEL_BLOCK in node else None
  if block and isinstance(block, transformers.SeriesWithResiduals):
    if len(block.blocks) == 2:
      attn, mlp = block.blocks
      if isinstance(
          attn, (transformers.AttentionHead, transformers.MultiAttentionHead)
      ) and isinstance(mlp, transformers.MLP):
        return True
  return False


def _all_attn_nodes(node_list: Sequence[Node]) -> bool:
  """Returns True iff all nodes are attention layers (or nodes is empty)."""
  for node in node_list:
    if not _node_is_attn(node):
      return False
  return True


def _all_mlp_nodes(node_list: Sequence[Node]) -> bool:
  """Returns True iff all nodes are MLP layers (or nodes is empty)."""
  for node in node_list:
    if not _node_is_mlp(node):
      return False
  return True


def _allocate_modules_to_layers(
    graph: nx.DiGraph, sources: Sequence[Node]
) -> Dict[int, int]:
  """Allocate all nodes in compute graph to layers.

  First, computes the longest path from the input to each node that is a model
  component (not input and output nodes). The longest path to a model component
  (its "depth") determines a layer in which we can place it while ensuring that
  all necessary previous computations have already happened.

  This assumes layers are arranged as [Attention, MLP, Attention, MLP, ...]

  In the special case where there are only Attention layers at one depth level
  and only MLP layers in the next depth layer, they are treated as if there
  are at the same depth because attention layers always come before MLP layers
  for the same depth.

  Args:
    graph: RASP graph with craft blocks.
    sources: List of input nodes

  Returns:
    A dict mapping from node ids to layer indices, where 0, 1, 2, 3, ...
    are in the order attention, mlp, attention, mlp, ...
  """
  layer_allocation: Dict[int, int] = collections.defaultdict(lambda: -1)
  depth_by_node_id: Dict[int, int] = dict()
  nodes_by_depth: Dict[int, List[Node]] = collections.defaultdict(list)
  computational_depth = compute_computational_depth(
      graph, [src[nodes.ID] for src in sources]
  )
  for node_id, node in graph.nodes.items():
    if (
        _node_is_mlp(node)
        or _node_is_attn(node)
        or _node_is_residual_block(node)
    ):
      # Node is a model component
      depth = computational_depth[node_id]
      depth_by_node_id[node_id] = depth
      nodes_by_depth[depth].append(node)

  # If at level `depth` there are only attention heads and at level `depths + 1`
  # there are only MLPs, we can condense them into one level
  # TODO(b/255936816): Think about improving this heuristic. The heuristic is
  # not optimal, and only catches very basic opportunities for optimization. It
  # is easy to come up with opportunities for optimization that it does not
  # catch.
  min_depth, max_depth = min(nodes_by_depth.keys()), max(nodes_by_depth.keys())
  depth = min_depth
  while depth < max_depth:
    if _all_attn_nodes(nodes_by_depth[depth]) and _all_mlp_nodes(
        nodes_by_depth[depth + 1]
    ):
      # Condense by decrementing the depth of all nodes starting from depth+1
      for update_depth in range(depth + 1, max_depth + 1):
        for node in nodes_by_depth[update_depth]:
          node_id = node[nodes.ID]
          depth_by_node_id[node_id] = update_depth - 1
        nodes_by_depth[update_depth - 1].extend(nodes_by_depth[update_depth])
        nodes_by_depth[update_depth] = []
      max_depth -= 1
    depth += 1

  # Allocate nodes to layers by depth, ensuring attn -> mlp -> attn -> mlp ...
  current_layer = 0
  current_depth = 1
  for node_id, depth in sorted(depth_by_node_id.items(), key=lambda x: x[1]):
    while depth > current_depth:
      current_depth += 1
      current_layer += 2
    if depth == current_depth:
      if _node_is_residual_block(graph.nodes[node_id]):
        layer_allocation[node_id] = current_layer
      else:
        is_mlp = _node_is_mlp(graph.nodes[node_id])
        layer_allocation[node_id] = current_layer + int(is_mlp)

  return layer_allocation


def craft_graph_to_model(
    graph: nx.DiGraph, sources: Sequence[Node]
) -> transformers.SeriesWithResiduals:
  """Translates a RASP graph with craft blocks into a full craft model.

  1. Allocate modules to layers, assuming layers in the order
  2. Creates subspaces for all inputs and outputs, and builds residual stream.
  3. Assembles everything into a craft model and returns it.

  Args:
    graph: RASP graph with craft blocks.
    sources: List of input nodes

  Returns:
    A craft model that can be compiled to model weights.

  Raises:
    ValueError: On invalid input (if the craft_graph does not have craft blocks
      already specified)
  """
  layer_allocation = _allocate_modules_to_layers(graph, sources)
  blocks_by_layer = collections.defaultdict(list)
  model_blocks = []

  residual_space = bases.VectorSpaceWithBasis([])

  for node_id, layer_no in layer_allocation.items():
    node = graph.nodes[node_id]
    block = node[nodes.MODEL_BLOCK] if nodes.MODEL_BLOCK in node else None

    if _node_is_residual_block(node):
      assert isinstance(block, transformers.SeriesWithResiduals)
      assert len(block.blocks) == 2
      residual_space = bases.join_vector_spaces(
          residual_space,
          block.blocks[0].residual_space,
          block.blocks[1].residual_space,
      )
      blocks_by_layer[layer_no].append(block.blocks[0])
      blocks_by_layer[layer_no + 1].append(block.blocks[1])
    elif block:
      residual_space = bases.join_vector_spaces(
          residual_space, node[nodes.MODEL_BLOCK].residual_space
      )
      blocks_by_layer[layer_no].append(block)

  for layer_no, layer_blocks in sorted(
      blocks_by_layer.items(), key=lambda x: x[0]
  ):
    for block in layer_blocks:
      block.residual_space = residual_space

    if layer_blocks:
      if layer_no % 2 == 0:  # Attention Layer
        multi_head_attn = transformers.MultiAttentionHead(layer_blocks)
        model_blocks.append(multi_head_attn)
      else:  # MLP Layer
        parallel_mlp = transformers.MLP.combine_in_parallel(layer_blocks)
        model_blocks.append(parallel_mlp)

  return transformers.SeriesWithResiduals(model_blocks)
