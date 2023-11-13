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
"""Combines all steps of compiling a RASP program."""

from typing import Set

from tracr.compiler import assemble
from tracr.compiler import basis_inference
from tracr.compiler import craft_graph_to_model
from tracr.compiler import craft_model_to_transformer
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import rasp_to_graph
from tracr.compiler import validating
from tracr.craft import bases
from tracr.rasp import rasp


COMPILER_BOS = "compiler_bos"
COMPILER_PAD = "compiler_pad"


def compile_rasp_to_model(
    program: rasp.SOp,
    vocab: Set[rasp.Value],
    max_seq_len: int,
    causal: bool = False,
    compiler_bos: str = COMPILER_BOS,
    compiler_pad: str = COMPILER_PAD,
    mlp_exactness: int = 100,
) -> assemble.AssembledTransformerModel:
  """Compile a RASP program to transformer weights.

  Note that currently not all RASP features are supported. Most unsupported
  features are detected at compile time and will cause a NotImplementedError.
  However, a few unsupported features cannot be checked at compile time and
  can cause silent errors.

  See `compiler.validating` for details and a function to quickly check if
  a program is compilable with Tracr without needing to compile it.

  Args:
    program: the RASP program to compile.
    vocab: the set of vocab tokens expected by RASP.
    max_seq_len: the maximum sequence length for the compiled model.
    causal: if True, outputs a model with causal masking.
    compiler_bos: the name of the special BOS token that will be added by the
      compiler. Must not be present in the vocab.
    compiler_pad: the name of the special PAD token that will be added by the
      compiler. Must not be present in the vocab.
    mlp_exactness: Controls the approximation of the MLP layers. In theory,
      larger values yield a better approximation. But too large values can cause
      numerical issues due to large parameter norms. Reasonable values are
      between 1 and 100.

  Returns:
    The compiled model.

  Raises:
    NotImplementedError: if the program uses unsopported features that can be
      caught at compile time.
  """

  if compiler_bos in vocab:
    raise ValueError(
        "Compiler BOS token must not be present in the vocab. "
        f"Found '{compiler_bos}' in {vocab}"
    )

  if compiler_pad in vocab:
    raise ValueError(
        "Compiler PAD token must not be present in the vocab. "
        f"Found '{compiler_pad}' in {vocab}"
    )

  # Perform static validation to fail fast. This catches most programs that
  # tracr is unable to compile.
  unsupported_exprs = validating.static_validate(program)
  if unsupported_exprs:
    error_message = "\n".join(
        (f"{expr.expr.name}: {expr.reason}" for expr in unsupported_exprs)
    )
    error_message = f"Unsupported RASP expressions:\n{error_message}"
    raise NotImplementedError(error_message)

  extracted = rasp_to_graph.extract_rasp_graph(program)
  graph, sources, sink = extracted.graph, extracted.sources, extracted.sink

  basis_inference.infer_bases(
      graph,
      sink,
      vocab,
      max_seq_len,
  )

  expr_to_craft_graph.add_craft_components_to_rasp_graph(
      graph,
      bos_dir=bases.BasisDirection(rasp.tokens.label, compiler_bos),
      mlp_exactness=mlp_exactness,
  )

  craft_model = craft_graph_to_model.craft_graph_to_model(graph, sources)

  return craft_model_to_transformer.craft_model_to_transformer(
      craft_model=craft_model,
      graph=graph,
      sink=sink,
      max_seq_len=max_seq_len,
      causal=causal,
      compiler_bos=compiler_bos,
      compiler_pad=compiler_pad,
  )
