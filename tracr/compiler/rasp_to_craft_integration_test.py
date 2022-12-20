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
"""Integration tests for the RASP -> craft stages of the compiler."""

import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tracr.compiler import basis_inference
from tracr.compiler import craft_graph_to_model
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import nodes
from tracr.compiler import rasp_to_graph
from tracr.compiler import test_cases
from tracr.craft import bases
from tracr.craft import tests_common
from tracr.rasp import rasp

_BOS_DIRECTION = "rasp_to_transformer_integration_test_BOS"
_ONE_DIRECTION = "rasp_to_craft_integration_test_ONE"


def _make_input_space(vocab, max_seq_len):
  tokens_space = bases.VectorSpaceWithBasis.from_values("tokens", vocab)
  indices_space = bases.VectorSpaceWithBasis.from_values(
      "indices", range(max_seq_len))
  one_space = bases.VectorSpaceWithBasis.from_names([_ONE_DIRECTION])
  bos_space = bases.VectorSpaceWithBasis.from_names([_BOS_DIRECTION])
  input_space = bases.join_vector_spaces(tokens_space, indices_space, one_space,
                                         bos_space)

  return input_space


def _embed_input(input_seq, input_space):
  bos_vec = input_space.vector_from_basis_direction(
      bases.BasisDirection(_BOS_DIRECTION))
  one_vec = input_space.vector_from_basis_direction(
      bases.BasisDirection(_ONE_DIRECTION))
  embedded_input = [bos_vec + one_vec]
  for i, val in enumerate(input_seq):
    i_vec = input_space.vector_from_basis_direction(
        bases.BasisDirection("indices", i))
    val_vec = input_space.vector_from_basis_direction(
        bases.BasisDirection("tokens", val))
    embedded_input.append(i_vec + val_vec + one_vec)
  return bases.VectorInBasis.stack(embedded_input)


def _embed_output(output_seq, output_space, categorical_output):
  embedded_output = []
  output_label = output_space.basis[0].name
  for x in output_seq:
    if x is None:
      out_vec = output_space.null_vector()
    elif categorical_output:
      out_vec = output_space.vector_from_basis_direction(
          bases.BasisDirection(output_label, x))
    else:
      out_vec = x * output_space.vector_from_basis_direction(
          output_space.basis[0])
    embedded_output.append(out_vec)
  return bases.VectorInBasis.stack(embedded_output)


class CompilerIntegrationTest(tests_common.VectorFnTestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="map",
          program=rasp.categorical(rasp.Map(lambda x: x + 1, rasp.tokens))),
      dict(
          testcase_name="sequence_map",
          program=rasp.categorical(
              rasp.SequenceMap(lambda x, y: x + y, rasp.tokens, rasp.indices))),
      dict(
          testcase_name="sequence_map_with_same_input",
          program=rasp.categorical(
              rasp.SequenceMap(lambda x, y: x + y, rasp.tokens, rasp.tokens))),
      dict(
          testcase_name="select_aggregate",
          program=rasp.Aggregate(
              rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ),
              rasp.Map(lambda x: 1, rasp.tokens))))
  def test_rasp_program_and_craft_model_produce_same_output(self, program):
    vocab = {0, 1, 2}
    max_seq_len = 3

    extracted = rasp_to_graph.extract_rasp_graph(program)
    basis_inference.infer_bases(
        extracted.graph,
        extracted.sink,
        vocab,
        max_seq_len=max_seq_len,
    )
    expr_to_craft_graph.add_craft_components_to_rasp_graph(
        extracted.graph,
        bos_dir=bases.BasisDirection(_BOS_DIRECTION),
        one_dir=bases.BasisDirection(_ONE_DIRECTION),
    )
    model = craft_graph_to_model.craft_graph_to_model(extracted.graph,
                                                      extracted.sources)
    input_space = _make_input_space(vocab, max_seq_len)
    output_space = bases.VectorSpaceWithBasis(
        extracted.sink[nodes.OUTPUT_BASIS])

    for val in vocab:
      test_input = _embed_input([val], input_space)
      rasp_output = program([val])
      expected_output = _embed_output(
          output_seq=rasp_output,
          output_space=output_space,
          categorical_output=True)
      test_output = model.apply(test_input).project(output_space)
      self.assertVectorAllClose(
          tests_common.strip_bos_token(test_output), expected_output)

  @parameterized.named_parameters(*test_cases.TEST_CASES)
  def test_compiled_models_produce_expected_output(self, program, vocab,
                                                   test_input, expected_output,
                                                   max_seq_len, **kwargs):
    del kwargs
    categorical_output = rasp.is_categorical(program)

    extracted = rasp_to_graph.extract_rasp_graph(program)
    basis_inference.infer_bases(
        extracted.graph,
        extracted.sink,
        vocab,
        max_seq_len=max_seq_len,
    )
    expr_to_craft_graph.add_craft_components_to_rasp_graph(
        extracted.graph,
        bos_dir=bases.BasisDirection(_BOS_DIRECTION),
        one_dir=bases.BasisDirection(_ONE_DIRECTION),
    )
    model = craft_graph_to_model.craft_graph_to_model(extracted.graph,
                                                      extracted.sources)
    input_space = _make_input_space(vocab, max_seq_len)
    output_space = bases.VectorSpaceWithBasis(
        extracted.sink[nodes.OUTPUT_BASIS])
    if not categorical_output:
      self.assertLen(output_space.basis, 1)

    test_input_vector = _embed_input(test_input, input_space)
    expected_output_vector = _embed_output(
        output_seq=expected_output,
        output_space=output_space,
        categorical_output=categorical_output)
    test_output = model.apply(test_input_vector).project(output_space)
    self.assertVectorAllClose(
        tests_common.strip_bos_token(test_output), expected_output_vector)

  @unittest.expectedFailure
  def test_setting_default_values_can_lead_to_wrong_outputs_in_compiled_model(
      self, program):
    # This is an example program in which setting a default value for aggregate
    # writes a value to the bos token position, which interfers with a later
    # aggregate operation causing the compiled model to have the wrong output.

    vocab = {"a", "b"}
    test_input = ["a"]
    max_seq_len = 2

    # RASP: [False, True]
    # compiled: [False, False, True]
    not_a = rasp.Map(lambda x: x != "a", rasp.tokens)

    # RASP:
    # [[True, False],
    #  [False, False]]
    # compiled:
    # [[False,True, False],
    #  [True, False, False]]
    sel1 = rasp.Select(rasp.tokens, rasp.tokens,
                       lambda k, q: k == "a" and q == "a")

    # RASP: [False, True]
    # compiled: [True, False, True]
    agg1 = rasp.Aggregate(sel1, not_a, default=True)

    # RASP:
    # [[False, True]
    #  [True, True]]
    # compiled:
    # [[True, False, False]
    #  [True, False, False]]
    # because pre-softmax we get
    # [[1.5, 1, 1]
    #  [1.5, 1, 1]]
    # instead of
    # [[0.5, 1, 1]
    #  [0.5, 1, 1]]
    # Because agg1 = True is stored on the BOS token position
    sel2 = rasp.Select(agg1, agg1, lambda k, q: k or q)

    # RASP: [1, 0.5]
    # compiled
    # [1, 1, 1]
    program = rasp.numerical(
        rasp.Aggregate(sel2, rasp.numerical(not_a), default=1))
    expected_output = [1, 0.5]

    # RASP program gives the correct output
    program_output = program(test_input)
    np.testing.assert_allclose(program_output, expected_output)

    extracted = rasp_to_graph.extract_rasp_graph(program)
    basis_inference.infer_bases(
        extracted.graph,
        extracted.sink,
        vocab,
        max_seq_len=max_seq_len,
    )
    expr_to_craft_graph.add_craft_components_to_rasp_graph(
        extracted.graph,
        bos_dir=bases.BasisDirection(_BOS_DIRECTION),
        one_dir=bases.BasisDirection(_ONE_DIRECTION),
    )
    model = craft_graph_to_model.craft_graph_to_model(extracted.graph,
                                                      extracted.sources)

    input_space = _make_input_space(vocab, max_seq_len)
    output_space = bases.VectorSpaceWithBasis(
        extracted.sink[nodes.OUTPUT_BASIS])

    test_input_vector = _embed_input(test_input, input_space)
    expected_output_vector = _embed_output(
        output_seq=expected_output,
        output_space=output_space,
        categorical_output=True)
    compiled_model_output = model.apply(test_input_vector).project(output_space)

    # Compiled craft model gives correct output
    self.assertVectorAllClose(
        tests_common.strip_bos_token(compiled_model_output),
        expected_output_vector)


if __name__ == "__main__":
  absltest.main()
