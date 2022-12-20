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
"""Tests for compiler.rasp_to_graph."""

from absl.testing import absltest
from absl.testing import parameterized
from tracr.compiler import nodes
from tracr.compiler import rasp_to_graph
from tracr.rasp import rasp


class ExtractRaspGraphTest(parameterized.TestCase):

  def test_primitives_have_no_edges(self):
    tokens_graph = rasp_to_graph.extract_rasp_graph(rasp.tokens).graph
    self.assertEmpty(tokens_graph.edges)

    indices_graph = rasp_to_graph.extract_rasp_graph(rasp.indices).graph
    self.assertEmpty(indices_graph.edges)

    full_graph = rasp_to_graph.extract_rasp_graph(rasp.Full(1)).graph
    self.assertEmpty(full_graph.edges)

  def test_one_edge(self):
    program = rasp.Map(lambda x: x + 1, rasp.tokens)

    graph = rasp_to_graph.extract_rasp_graph(program).graph

    self.assertLen(graph.edges, 1)
    (u, v), = graph.edges
    self.assertEqual(graph.nodes[u][nodes.EXPR], rasp.tokens)
    self.assertEqual(graph.nodes[v][nodes.EXPR], program)

  def test_aggregate(self):
    program = rasp.Aggregate(
        rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ),
        rasp.indices,
    )

    extracted = rasp_to_graph.extract_rasp_graph(program)

    # Expected graph:
    #
    # indices \ --------
    #          \         \
    #           select -- program
    # tokens  /

    self.assertLen(extracted.graph.edges, 4)
    self.assertEqual(extracted.sink[nodes.EXPR], program)
    for source in extracted.sources:
      self.assertIn(
          source[nodes.EXPR],
          [rasp.tokens, rasp.indices],
      )


if __name__ == "__main__":
  absltest.main()
