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
"""Tests for compiler.craft_graph_to_model."""

from absl.testing import absltest
from absl.testing import parameterized
import networkx as nx
from tracr.compiler import craft_graph_to_model
from tracr.compiler import nodes
from tracr.compiler import rasp_to_graph
from tracr.craft import bases
from tracr.craft.chamber import categorical_attn
from tracr.craft.chamber import categorical_mlp
from tracr.rasp import rasp


class CraftAllocateModulesToLayersTest(parameterized.TestCase):

  def _get_dummy_block(self, block_type):
    if block_type == "ATTN":
      return categorical_attn.categorical_attn(
          query_space=bases.VectorSpaceWithBasis.from_names(["query"]),
          key_space=bases.VectorSpaceWithBasis.from_names(["bos", "key"]),
          value_space=bases.VectorSpaceWithBasis.from_names(["bos", "value"]),
          output_space=bases.VectorSpaceWithBasis.from_names(["output"]),
          bos_space=bases.VectorSpaceWithBasis.from_names(["bos"]),
          one_space=bases.VectorSpaceWithBasis.from_names(["one"]),
          attn_fn=lambda x, y: True,
      )
    elif block_type == "MLP":
      return categorical_mlp.map_categorical_mlp(
          input_space=bases.VectorSpaceWithBasis.from_names(["input"]),
          output_space=bases.VectorSpaceWithBasis.from_names(["output"]),
          operation=lambda x: x,
      )
    else:
      return None

  def test_compute_computational_depth_returns_expected_result(self):
    """Creates a graph and checks the longest path for each node."""

    # Node IDs:
    # 0 -- 1 -- 2 -- 3 ------------  4
    #               /              /
    # 5 -- 6 ---------- 7 -- 8 -- 9
    #
    # 10
    # Expected return values:
    # 0 -- 1 -- 2 -- 3 ------------  5
    #               /              /
    # 0 -- 1 ---------- 2 -- 3 -- 4
    #
    # -1

    graph = nx.DiGraph()
    node_ids = list(range(11))
    expected_results = [0, 1, 2, 3, 5, 0, 1, 2, 3, 4, -1]
    for node_id, res in zip(node_ids, expected_results):
      graph.add_node(
          node_id, **{
              nodes.ID: node_id,
              nodes.EXPR: rasp.ConstantSOp(1),
              "expected_result": res
          })
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(5, 6)
    graph.add_edge(6, 7)
    graph.add_edge(7, 8)
    graph.add_edge(8, 9)
    graph.add_edge(6, 3)
    graph.add_edge(9, 4)
    sources = [graph.nodes[0], graph.nodes[5]]

    computational_depth = craft_graph_to_model.compute_computational_depth(
        graph, [src[nodes.ID] for src in sources]
    )
    for node_id, node in graph.nodes.items():
      self.assertEqual(computational_depth[node_id], node["expected_result"])

  def test_allocate_modules_to_layers_returns_expected_result(self):
    """Creates a graph and checks if the correct layer assignment is returned."""

    # Computation Graph:
    # INPUT -- ATTN -- MLP -- ATTN ------ MLP -- OUTPUT
    #           /           /          /
    # INPUT -- MLP --- MLP          ATTN
    #                      \      /
    #                        ATTN
    # Node IDs:
    # 0 -- 1 -- 2 -- 3 -- 4 -- 5
    #         /     /     /
    # 6 -- 7 ---- 8      9
    #               \   /
    #                10
    # Expected layer allocation:
    # -1 -- 0 -- 3 -- 4 -- 7 -- -1
    #         /     /     /
    # -1 -- 1 --- 3      6
    #               \   /
    #                 4

    graph = nx.DiGraph()
    node_ids = list(range(11))
    types = [
        "INPUT", "ATTN", "MLP", "ATTN", "MLP", "OUTPUT", "INPUT", "MLP", "MLP",
        "ATTN", "ATTN"
    ]
    expected_results = [-1, 0, 3, 4, 7, -1, -1, 1, 3, 6, 4]
    for node_id, node_type, res in zip(node_ids, types, expected_results):
      graph.add_node(
          node_id, **{
              nodes.ID: node_id,
              nodes.EXPR: rasp.ConstantSOp(1),
              nodes.MODEL_BLOCK: self._get_dummy_block(node_type),
              "expected_result": res
          })

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(6, 7)
    graph.add_edge(7, 2)
    graph.add_edge(7, 8)
    graph.add_edge(8, 3)
    graph.add_edge(8, 10)
    graph.add_edge(9, 4)
    graph.add_edge(10, 9)

    craft_graph = rasp_to_graph.ExtractRaspGraphOutput(
        graph=graph,
        sink=graph.nodes[10],
        sources=[graph.nodes[0], graph.nodes[6]])

    layer_allocation = craft_graph_to_model._allocate_modules_to_layers(
        craft_graph.graph, craft_graph.sources)
    for node_id, node in graph.nodes.items():
      self.assertEqual(layer_allocation[node_id], node["expected_result"])

  def test_allocate_modules_to_layers_returns_expected_result_for_chain(self):
    """Tests a chain of alternating attention layers and MLPs."""

    # Computation Graph:
    # INPUT -- ATTN -- MLP -- ATTN -- MLP -- OUTPUT
    # Node IDs:
    # 0 -- 1 -- 2 -- 3 -- 4 -- 5
    # Expected layer allocation:
    # -1 -- 0 -- 1 -- 2 -- 3 -- -1

    graph = nx.DiGraph()
    node_ids = list(range(11))
    types = ["INPUT", "ATTN", "MLP", "ATTN", "MLP", "OUTPUT"]
    expected_results = [-1, 0, 1, 2, 3, -1]
    for node_id, node_type, res in zip(node_ids, types, expected_results):
      graph.add_node(
          node_id, **{
              nodes.ID: node_id,
              nodes.EXPR: rasp.ConstantSOp(1),
              nodes.MODEL_BLOCK: self._get_dummy_block(node_type),
              "expected_result": res
          })

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)

    craft_graph = rasp_to_graph.ExtractRaspGraphOutput(
        graph=graph, sink=graph.nodes[5], sources=[graph.nodes[0]])

    layer_allocation = craft_graph_to_model._allocate_modules_to_layers(
        craft_graph.graph, craft_graph.sources)
    for node_id, node in graph.nodes.items():
      self.assertEqual(layer_allocation[node_id], node["expected_result"])


if __name__ == "__main__":
  absltest.main()
