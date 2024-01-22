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
"""Convert craft model into transformer with the correct input/output spaces."""

import networkx as nx
from tracr.compiler import assemble
from tracr.compiler import nodes
from tracr.craft import bases
from tracr.craft import transformers
from tracr.rasp import rasp
from tracr.transformer import encoder


def craft_model_to_transformer(
    craft_model: transformers.SeriesWithResiduals,
    graph: nx.DiGraph,
    sink: nodes.Node,
    max_seq_len: int,
    compiler_bos: str,
    compiler_pad: str,
    causal: bool = False,
) -> assemble.AssembledTransformerModel:
  """Turn a craft model into a transformer model."""

  if rasp.tokens.label not in graph.nodes:
    raise ValueError(
        f'Failed to find a node with label {rasp.tokens.label}. '
        'This is probably because your RASP program does not include '
        'rasp.tokens. A program must include rasp.tokens to be '
        'compiled.'
    )

  # Add the compiler BOS token.
  tokens_value_set = (
      graph.nodes[rasp.tokens.label][nodes.VALUE_SET].union(
          {compiler_bos, compiler_pad}))
  tokens_space = bases.VectorSpaceWithBasis.from_values(rasp.tokens.label,
                                                        tokens_value_set)

  indices_space = bases.VectorSpaceWithBasis.from_values(
      rasp.indices.label, range(max_seq_len))

  categorical_output = rasp.is_categorical(sink[nodes.EXPR])
  output_space = bases.VectorSpaceWithBasis(sink[nodes.OUTPUT_BASIS])

  assembled_model = assemble.assemble_craft_model(
      craft_model=craft_model,
      tokens_space=tokens_space,
      indices_space=indices_space,
      output_space=output_space,
      categorical_output=categorical_output,
      causal=causal,
  )

  assembled_model.input_encoder = encoder.CategoricalEncoder(
      basis=tokens_space.basis,
      enforce_bos=compiler_bos is not None,
      bos_token=compiler_bos,
      pad_token=compiler_pad,
      max_seq_len=max_seq_len + 1 if compiler_bos is not None else max_seq_len,
  )

  if categorical_output:
    assembled_model.output_encoder = encoder.CategoricalEncoder(
        basis=output_space.basis,
        enforce_bos=False,
        bos_token=None,
        pad_token=None)
  else:
    assembled_model.output_encoder = encoder.NumericalEncoder()

  return assembled_model
