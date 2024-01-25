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
"""Add craft model blocks to graph of RASPExpr."""

from typing import Any, Callable, Optional

import networkx as nx
from tracr.compiler import nodes
from tracr.craft import bases
from tracr.craft.chamber import categorical_attn
from tracr.craft.chamber import categorical_mlp
from tracr.craft.chamber import numerical_mlp
from tracr.craft.chamber import selector_width
from tracr.rasp import rasp


def _transform_fun_to_basis_fun(
    fun: Callable[..., Any],
    output_direction_name: Optional[str] = None) -> Callable[..., Any]:
  """Transforms a function acting on values into one acting on directions."""

  def bases_fun(*args):
    values = [d.value for d in args]
    result = fun(*values)
    if output_direction_name:
      return bases.BasisDirection(output_direction_name, result)
    return result

  return bases_fun


def _check_selector_expression(expr, graph):
  """Check graph structure and encodings for an aggregate or selector width."""
  sel_expr = expr.selector

  # Check graph structure
  assert sel_expr.label in graph.predecessors(expr.label)
  assert sel_expr.keys.label in graph.predecessors(sel_expr.label)
  assert sel_expr.queries.label in graph.predecessors(sel_expr.label)

  if (not rasp.is_categorical(sel_expr.queries) or
      not rasp.is_categorical(sel_expr.keys)):
    raise NotImplementedError("Selector keys and queries must be categorical.")


def add_craft_components_to_rasp_graph(
    graph: nx.DiGraph,
    bos_dir: bases.BasisDirection = bases.BasisDirection("tokens", "bos"),
    one_dir: bases.BasisDirection = bases.BasisDirection("one"),
    causal: bool = False,
    mlp_exactness: float = 100,
) -> None:
  """Translates expressions to craft blocks and attaches them to the graph.

  Sets the `MODEL_BLOCK` attribute for all nodes in `graph`.

  Args:
    graph: RASP graph with  `VALUE_SET` but not `MODEL_BLOCK` attributes.
    bos_dir: Basis direction representing beginning of sequence (bos) token.
    one_dir: Auxiliary basis direction that must contain 1.
    causal: If True, marks attention blocks as causal.
    mlp_exactness: Controls the approximation of the MLP layers.

  Raises:
    ValueError: On invalid input (if `MODEL_BLOCK` is set already, or
      `VALUE_SET` is not set already)
    NotImplementedError: If the graph contains an unsupported expression.
  """
  one_space = bases.VectorSpaceWithBasis([one_dir])

  for node_id, node in graph.nodes.items():
    expr = node[nodes.EXPR]

    if not isinstance(expr, rasp.SOp):
      continue

    if nodes.MODEL_BLOCK in node and node[nodes.MODEL_BLOCK]:
      raise ValueError("Input graph cannot have model blocks set already.")
    if nodes.VALUE_SET not in node:
      raise ValueError(
          "Craft components can only be added after basis inference.")

    if isinstance(expr, (rasp.TokensType, rasp.IndicesType)):
      block = None
    elif isinstance(expr, rasp.Map):
      inner_expr, inner_node = expr.inner, graph.nodes[expr.inner.label]
      assert inner_expr.label in graph.predecessors(node_id)
      input_space = bases.VectorSpaceWithBasis(inner_node[nodes.OUTPUT_BASIS])
      output_space = bases.VectorSpaceWithBasis(node[nodes.OUTPUT_BASIS])

      if rasp.is_categorical(inner_expr) and rasp.is_categorical(expr):
        basis_fun = _transform_fun_to_basis_fun(expr.f, expr.label)
        block = categorical_mlp.map_categorical_mlp(
            input_space=input_space,
            output_space=output_space,
            operation=basis_fun)
      elif rasp.is_categorical(inner_expr) and rasp.is_numerical(expr):
        block = categorical_mlp.map_categorical_to_numerical_mlp(
            input_space=input_space,
            output_space=output_space,
            operation=expr.f,
        )
      elif rasp.is_numerical(inner_expr) and rasp.is_categorical(expr):
        block = numerical_mlp.map_numerical_to_categorical_mlp(
            f=expr.f,
            input_space=input_space,
            output_space=output_space,
            input_value_set=inner_node[nodes.VALUE_SET],
            one_space=one_space,
            hidden_name=f"_hidden_{expr.label}_",
            large_number=mlp_exactness)
      elif rasp.is_numerical(inner_expr) and rasp.is_numerical(expr):
        block = numerical_mlp.map_numerical_mlp(
            f=expr.f,
            input_space=input_space,
            output_space=output_space,
            input_value_set=inner_node[nodes.VALUE_SET],
            one_space=one_space,
            hidden_name=f"_hidden_{expr.label}_",
            large_number=mlp_exactness)
      else:
        raise NotImplementedError("Map does no support "
                                  f"in_type '{inner_expr.type}' and"
                                  f" out_type '{expr.type}'!")

    elif isinstance(expr, rasp.SequenceMap):
      fst_expr, fst_node = expr.fst, graph.nodes[expr.fst.label]
      snd_expr, snd_node = expr.snd, graph.nodes[expr.snd.label]

      # Check graph structure
      assert fst_expr.label in graph.predecessors(node_id)
      assert snd_expr.label in graph.predecessors(node_id)

      fst_space = bases.VectorSpaceWithBasis(fst_node[nodes.OUTPUT_BASIS])
      snd_space = bases.VectorSpaceWithBasis(snd_node[nodes.OUTPUT_BASIS])
      out_space = bases.VectorSpaceWithBasis(node[nodes.OUTPUT_BASIS])

      if (isinstance(expr, rasp.LinearSequenceMap) and
          not all(rasp.is_numerical(x) for x in (fst_expr, snd_expr, expr))):
        raise NotImplementedError("Linear SequenceMap only supports numerical "
                                  "inputs/outputs.")
      elif (
          not isinstance(expr, rasp.LinearSequenceMap) and
          not all(rasp.is_categorical(x) for x in (fst_expr, snd_expr, expr))):
        raise NotImplementedError("(Non-linear) SequenceMap only supports "
                                  "categorical inputs/outputs.")

      if isinstance(expr, rasp.LinearSequenceMap):
        assert len(fst_space.basis) == 1
        assert len(snd_space.basis) == 1
        assert len(out_space.basis) == 1
        block = numerical_mlp.linear_sequence_map_numerical_mlp(
            input1_basis_direction=fst_space.basis[0],
            input2_basis_direction=snd_space.basis[0],
            output_basis_direction=out_space.basis[0],
            input1_factor=expr.fst_fac,
            input2_factor=expr.snd_fac,
            hidden_name=f"_hidden_{expr.label}_")
      elif fst_space == snd_space:
        # It's okay to use the local variable expr.f because it is
        # only used within the same loop iteration to create the MLP.
        # pylint: disable=cell-var-from-loop
        basis_fun = _transform_fun_to_basis_fun(lambda x: expr.f(x, x),
                                                expr.label)
        block = categorical_mlp.map_categorical_mlp(
            input_space=fst_space, output_space=out_space, operation=basis_fun)
      else:
        basis_fun = _transform_fun_to_basis_fun(expr.f, expr.label)
        block = categorical_mlp.sequence_map_categorical_mlp(
            input1_space=fst_space,
            input2_space=snd_space,
            output_space=out_space,
            operation=basis_fun,
            one_space=one_space,
            hidden_name=f"_hidden_{expr.label}_")
    elif isinstance(expr, rasp.Aggregate):
      sel_expr: rasp.Select = expr.selector
      agg_expr: rasp.Aggregate = expr

      if not isinstance(sel_expr, rasp.Select):
        raise TypeError("Compiling composite Selectors is not supported. "
                        f"Got a {sel_expr}.")

      queries = graph.nodes[sel_expr.queries.label]
      keys = graph.nodes[sel_expr.keys.label]
      sop = graph.nodes[agg_expr.sop.label]

      _check_selector_expression(expr, graph)
      assert agg_expr.sop.label in graph.predecessors(node_id)
      if rasp.get_encoding(agg_expr.sop) != rasp.get_encoding(agg_expr):
        raise NotImplementedError(
            "An Aggregate's output encoding must match the input encoding."
            f" Input: {rasp.get_encoding(agg_expr.sop)},"
            f" Output: {rasp.get_encoding(agg_expr)}"
        )
      if rasp.is_categorical(agg_expr) and agg_expr.default is not None:
        raise NotImplementedError(
            "Default for a categorical aggregate must be None. "
            f"Got {agg_expr.default}"
        )
      if rasp.is_numerical(agg_expr) and agg_expr.default != 0:
        raise NotImplementedError(
            "Default for a numerical aggregate must be 0. "
            f"Got {agg_expr.default}"
        )

      bos_space = bases.VectorSpaceWithBasis([bos_dir])
      one_space = bases.VectorSpaceWithBasis([one_dir])
      query_space = bases.VectorSpaceWithBasis(queries[nodes.OUTPUT_BASIS])
      key_space = bases.VectorSpaceWithBasis(keys[nodes.OUTPUT_BASIS])
      value_space = bases.VectorSpaceWithBasis(sop[nodes.OUTPUT_BASIS])
      output_space = bases.VectorSpaceWithBasis(node[nodes.OUTPUT_BASIS])

      # Argument order is different in craft / transformers than RASP selectors
      def attn_basis_fn(query: bases.BasisDirection,
                        key: bases.BasisDirection) -> bool:
        # It's okay to use the local variable sel_expr because this function is
        # only used within the same loop iteration to create an attention head.
        # pylint: disable=cell-var-from-loop
        selector_basis_fn = _transform_fun_to_basis_fun(sel_expr.predicate)
        return selector_basis_fn(key, query)

      block = categorical_attn.categorical_attn(
          query_space=query_space,
          key_space=key_space,
          value_space=value_space,
          output_space=output_space,
          bos_space=bos_space,
          one_space=one_space,
          attn_fn=attn_basis_fn,
          default_output=output_space.null_vector(),
          causal=causal,
          always_attend_to_bos=False,
          use_bos_for_default_output=True,
          softmax_coldness=100)
    elif isinstance(expr, rasp.SelectorWidth):
      sel_expr = expr.selector
      queries = graph.nodes[sel_expr.queries.label]
      keys = graph.nodes[sel_expr.keys.label]
      _check_selector_expression(expr, graph)

      bos_space = bases.VectorSpaceWithBasis([bos_dir])
      query_space = bases.VectorSpaceWithBasis(queries[nodes.OUTPUT_BASIS])
      key_space = bases.VectorSpaceWithBasis(keys[nodes.OUTPUT_BASIS])
      output_space = bases.VectorSpaceWithBasis(node[nodes.OUTPUT_BASIS])

      # Argument order is different in craft / transformers than RASP selectors
      def attn_basis_fn(query: bases.BasisDirection,
                        key: bases.BasisDirection) -> bool:
        # It's okay to use the local variable sel_expr because this function is
        # only used within the same loop iteration to create an attention head.
        selector_basis_fn = _transform_fun_to_basis_fun(sel_expr.predicate)  # pylint: disable=cell-var-from-loop
        return selector_basis_fn(key, query)

      block = selector_width.selector_width(
          query_space=query_space,
          key_space=key_space,
          output_space=output_space,
          bos_space=bos_space,
          one_space=one_space,
          attn_fn=attn_basis_fn,
          out_value_set=node[nodes.VALUE_SET],
          categorical_output=rasp.is_categorical(expr),
          causal=False,
          softmax_coldness=100,
          mlp_large_number=mlp_exactness,
          label=expr.label)
    else:
      raise NotImplementedError(f"Expression {expr} cannot be translated to "
                                "a model component.")

    graph.nodes[node_id][nodes.MODEL_BLOCK] = block
