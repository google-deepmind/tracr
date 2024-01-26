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
"""Tests for compiler.expr_to_craft_graph."""

from absl.testing import absltest
from absl.testing import parameterized
from tracr.compiler import basis_inference
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import lib
from tracr.compiler import nodes
from tracr.compiler import rasp_to_graph
from tracr.craft import bases
from tracr.craft import transformers
from tracr.rasp import rasp


class ExprToCraftGraphTest(parameterized.TestCase):

  def _check_block_types_are_correct(self, graph):
    for _, node in graph.nodes.items():
      expr = node[nodes.EXPR]
      if isinstance(expr, rasp.SOp):
        block = node[nodes.MODEL_BLOCK]
        if isinstance(expr, (rasp.Map, rasp.SequenceMap)):
          self.assertIsInstance(block, transformers.MLP)
        elif isinstance(expr, rasp.Aggregate):
          self.assertIsInstance(block, transformers.AttentionHead)

  def _get_input_space_from_node(self, node):
    block = node[nodes.MODEL_BLOCK]
    if isinstance(block, transformers.MLP):
      return block.fst.input_space
    elif isinstance(block, transformers.AttentionHead):
      return bases.join_vector_spaces(block.w_qk.left_space,
                                      block.w_qk.right_space,
                                      block.w_ov.input_space)
    else:
      return None

  def _check_spaces_are_consistent(self, graph):
    """Check that for each edge the output is a subspace of the input."""
    for u, v in graph.edges:
      u_node, v_node = graph.nodes[u], graph.nodes[v]
      if isinstance(u_node[nodes.EXPR], rasp.SOp) and isinstance(
          v_node[nodes.EXPR], rasp.SOp):
        u_out_basis = u_node[nodes.OUTPUT_BASIS]
        u_out_space = bases.VectorSpaceWithBasis(u_out_basis)
        v_in_space = self._get_input_space_from_node(v_node)
        self.assertTrue(u_out_space.issubspace(v_in_space))

  @parameterized.named_parameters(
      dict(
          testcase_name="single_map",
          program=rasp.Map(lambda x: x + 1, rasp.tokens),
      ),
      dict(
          testcase_name="single_sequence_map",
          program=rasp.SequenceMap(
              lambda x, y: x + y, rasp.tokens, rasp.indices
          ),
      ),
      dict(
          testcase_name="single_select_aggregate",
          program=rasp.Aggregate(
              rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ),
              rasp.tokens,
          ),
      ),
      dict(testcase_name="reverse", program=lib.make_reverse(rasp.tokens)),
      dict(testcase_name="length", program=lib.make_length()),
      dict(
          testcase_name="annotated_tokens",
          program=rasp.annotate(rasp.tokens, foo="foo"),
      ),
      dict(
          testcase_name="annotated_indices",
          program=rasp.annotate(rasp.indices, foo="foo"),
      ),
  )
  def test_compiling_rasp_programs(self, program):
    vocab = {0, 1, 2}
    extracted = rasp_to_graph.extract_rasp_graph(program)
    basis_inference.infer_bases(
        extracted.graph,
        extracted.sink,
        vocab,
        max_seq_len=3,
    )
    expr_to_craft_graph.add_craft_components_to_rasp_graph(extracted.graph)
    self._check_block_types_are_correct(extracted.graph)
    self._check_spaces_are_consistent(extracted.graph)

  def test_add_craft_components_raises_value_error_if_called_before_basis_inference(
      self):
    program = rasp.categorical(rasp.Map(lambda x: x + 1, rasp.tokens))
    extracted = rasp_to_graph.extract_rasp_graph(program)

    with self.assertRaisesRegex(
        ValueError,
        r"^.*Craft components can only be added after basis inference.*$"):
      expr_to_craft_graph.add_craft_components_to_rasp_graph(extracted.graph)

  def test_add_craft_components_raises_value_error_if_called_twice(self):
    vocab = {0, 1, 2}
    program = rasp.categorical(rasp.Map(lambda x: x + 1, rasp.tokens))
    extracted = rasp_to_graph.extract_rasp_graph(program)

    basis_inference.infer_bases(
        extracted.graph,
        extracted.sink,
        vocab,
        max_seq_len=1,
    )

    expr_to_craft_graph.add_craft_components_to_rasp_graph(extracted.graph)
    with self.assertRaisesRegex(
        ValueError, r"^.*Input graph cannot have model blocks set already.*$"):
      expr_to_craft_graph.add_craft_components_to_rasp_graph(extracted.graph)


if __name__ == "__main__":
  absltest.main()
