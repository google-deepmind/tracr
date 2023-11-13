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
"""Tests for compiler.basis_inference."""

from absl.testing import absltest
from absl.testing import parameterized
from tracr.compiler import basis_inference
from tracr.compiler import nodes
from tracr.compiler import rasp_to_graph
from tracr.rasp import rasp


class InferBasesTest(parameterized.TestCase):

  def test_arithmetic_error_logs_warning(self):
    program = rasp.numerical(rasp.Map(lambda x: 1 / x, rasp.tokens))
    extracted = rasp_to_graph.extract_rasp_graph(program)
    vocab = {0, 1, 2}
    with self.assertLogs(level="WARNING"):
      basis_inference.infer_bases(
          extracted.graph,
          extracted.sink,
          vocab,
          max_seq_len=1,
      )

  @parameterized.parameters(({1, 2, 3}, {2, 3, 4}), ({0, 5}, {1, 6}))
  def test_one_edge(self, vocab, expected_value_set):
    program = rasp.categorical(rasp.Map(lambda x: x + 1, rasp.tokens))
    extracted = rasp_to_graph.extract_rasp_graph(program)

    basis_inference.infer_bases(
        extracted.graph,
        extracted.sink,
        vocab,
        max_seq_len=1,
    )

    self.assertSetEqual(
        extracted.graph.nodes[program.label][nodes.VALUE_SET],
        expected_value_set,
    )

  def test_primitive_close_to_tip(self):
    intermediate = rasp.categorical(rasp.tokens + 1)
    intermediate = rasp.categorical(intermediate + intermediate)
    program = rasp.categorical(intermediate + rasp.indices)
    extracted = rasp_to_graph.extract_rasp_graph(program)

    basis_inference.infer_bases(
        extracted.graph,
        extracted.sink,
        {0, 1},
        max_seq_len=2,
    )

    self.assertSetEqual(
        extracted.graph.nodes[program.label][nodes.VALUE_SET],
        {2, 3, 4, 5},
    )
    self.assertSetEqual(
        extracted.graph.nodes[intermediate.label][nodes.VALUE_SET],
        {2, 3, 4},
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="categorical_aggregate",
          program=rasp.categorical(
              rasp.Aggregate(
                  rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ),
                  rasp.indices,
              )
          ),
          vocab={0, 1},
          max_seq_len=3,
          expected_value_set={0, 1, 2},
      ),
      dict(
          testcase_name="numerical_aggregate",
          program=rasp.numerical(
              rasp.Aggregate(
                  rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ),
                  rasp.tokens,
              )
          ),
          vocab={0, 1},
          max_seq_len=2,
          expected_value_set={0, 1, 1 / 2},
      ),
      dict(
          testcase_name="selector_width",
          program=rasp.SelectorWidth(
              rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ)
          ),
          vocab={0, 1},
          max_seq_len=3,
          expected_value_set={0, 1, 2, 3},
      ),
      dict(
          testcase_name="annotated_tokens",
          program=rasp.categorical(rasp.tokens),
          vocab={"a", "b"},
          max_seq_len=2,
          expected_value_set={"a", "b"},
      ),
      dict(
          testcase_name="annotated_indices",
          program=rasp.categorical(rasp.indices),
          vocab={"a", "b"},
          max_seq_len=2,
          expected_value_set={0, 1},
      ),
  )
  def test_inferred_value_set_as_expected(
      self, program, vocab, max_seq_len, expected_value_set
  ):
    extracted = rasp_to_graph.extract_rasp_graph(program)

    basis_inference.infer_bases(
        extracted.graph,
        extracted.sink,
        vocab,
        max_seq_len=max_seq_len,
    )

    self.assertSetEqual(
        extracted.graph.nodes[program.label][nodes.VALUE_SET],
        expected_value_set,
    )


if __name__ == "__main__":
  absltest.main()
