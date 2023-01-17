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
"""Converting a RaspExpr to a graph."""

import dataclasses
import queue
from typing import List

import networkx as nx
from tracr.compiler import nodes
from tracr.rasp import rasp

Node = nodes.Node
NodeID = nodes.NodeID


@dataclasses.dataclass
class ExtractRaspGraphOutput:
  graph: nx.DiGraph
  sink: Node  # the program's output.
  sources: List[Node]  # the primitive S-Ops.


def extract_rasp_graph(tip: rasp.SOp) -> ExtractRaspGraphOutput:
  """Converts a RASP program into a graph representation."""
  expr_queue = queue.Queue()
  graph = nx.DiGraph()
  sources: List[NodeID] = []

  def ensure_node(expr: rasp.RASPExpr) -> NodeID:
    """Finds or creates a graph node corresponding to expr; returns its ID."""
    node_id = expr.label
    if node_id not in graph:
      graph.add_node(node_id, **{nodes.ID: node_id, nodes.EXPR: expr})

    return node_id

  # Breadth-first search over the RASP expression graph.

  def visit_raspexpr(expr: rasp.RASPExpr):
    parent_id = ensure_node(expr)

    for child_expr in expr.children:
      expr_queue.put(child_expr)
      child_id = ensure_node(child_expr)
      graph.add_edge(child_id, parent_id)

    if not expr.children:
      sources.append(graph.nodes[parent_id])

  expr_queue.put(tip)
  sink = graph.nodes[ensure_node(tip)]
  while not expr_queue.empty():
    visit_raspexpr(expr_queue.get())

  return ExtractRaspGraphOutput(graph=graph, sink=sink, sources=sources)
