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
"""RASP Evaluator which applies causal masks to selectors."""

import dataclasses
import queue
from typing import Sequence, Union

from tracr.rasp import rasp


@dataclasses.dataclass
class TracrUnsupportedExpr:
  """An uncompilable expression and the reason it's not compilable."""

  expr: rasp.RASPExpr
  reason: str


def _static_validate_expr(expr: rasp.RASPExpr) -> TracrUnsupportedExpr | None:
  """Returns TracrUnsupportedExpr if `expr` is not supported by Tracr."""
  if isinstance(expr, rasp.TokensType) and rasp.is_numerical(expr):
    return TracrUnsupportedExpr(
        expr=expr, reason="tokens should always be categorical."
    )
  if isinstance(expr, rasp.IndicesType) and rasp.is_numerical(expr):
    return TracrUnsupportedExpr(
        expr=expr, reason="tokens should always be categorical."
    )

  if isinstance(expr, rasp.Select):
    if not rasp.is_categorical(expr.keys):
      return TracrUnsupportedExpr(
          expr=expr,
          reason="Select keys must be categorical.",
      )
    if not rasp.is_categorical(expr.queries):
      return TracrUnsupportedExpr(
          expr=expr,
          reason="Select queries must be categorical.",
      )

  if isinstance(expr, rasp.Aggregate):
    if rasp.get_encoding(expr) != rasp.get_encoding(expr.sop):
      return TracrUnsupportedExpr(
          expr=expr,
          reason=(
              "An aggregate's output encoding must match its input encoding."
              f" Input: {rasp.get_encoding(expr)}  "
              f" Output: {rasp.get_encoding(expr.sop)}  "
          ),
      )

    if rasp.is_categorical(expr) and expr.default is not None:
      return TracrUnsupportedExpr(
          expr=expr,
          reason="Categorical aggregate only supports None as default value.",
      )
    if rasp.is_numerical(expr) and expr.default != 0:
      return TracrUnsupportedExpr(
          expr=expr,
          reason="Numerical aggregate only supports 0 as default value.",
      )

  if isinstance(expr, rasp.SequenceMap):
    if not isinstance(expr, rasp.LinearSequenceMap) and not all(
        rasp.is_categorical(x) for x in (expr.fst, expr.snd, expr)
    ):
      return TracrUnsupportedExpr(
          expr=expr,
          reason=(
              "(Non-linear) SequenceMap only supports categorical"
              " inputs/outputs."
          ),
      )

    if isinstance(expr, rasp.LinearSequenceMap) and not all(
        rasp.is_numerical(x) for x in (expr.fst, expr.snd, expr)
    ):
      return TracrUnsupportedExpr(
          expr=expr,
          reason="LinearSequenceMap only supports numerical inputs/outputs.",
      )


class DynamicValidationEvaluator(rasp.DefaultRASPEvaluator):
  """Evaluates RASP program but raises exceptions to anticipate compiler issues.

  Most features not supported by Tracr are specific input/output types for
  some SOp types and can be checked statically. For example, Tracr does not
  support Aggregate operations with different input and output encodings
  (instead, explicit conversion via a Map is required).

  There are some specific aggregate operations that are not supported and have
  to be checked dynamically. For example, Tracr does not support categorical
  Aggregate operations that require non-trival aggregation (eg, averaging
  tokens instead of moving tokens).
  """

  def __init__(self):
    self.unsupported_exprs = []
    super().__init__()

  def evaluate(
      self, expr: rasp.RASPExpr, xs: Sequence[rasp.Value]
  ) -> Union[Sequence[rasp.Value], rasp.SelectorValue]:
    out = super().evaluate(expr, xs)

    if isinstance(expr, rasp.Aggregate):
      # We support compiling programs which use Aggregates to move a single
      # categorical value to another position, ie when the attention pattern is
      # 1 in one place and 0 otherwise. However, if the attention pattern has
      # two or more 1s attending to different tokens that have to be aggregated
      # the compiled model will silently give incorrect outputs. We don't have
      # a way to do this statically so we have to check this at runtime.

      agg_in = expr.sop(xs)
      if (
          # The easiest way to satisfy this is to have a selector of width 1
          rasp.is_categorical(expr)
          and not set(out).issubset(set(agg_in) | {None})
      ):
        self.unsupported_exprs.append(
            TracrUnsupportedExpr(
                expr=expr,
                reason=(
                    "Categorical aggregate does not support Selectors with"
                    " width > 1 that require aggregation (eg. averaging)."
                ),
            )
        )
      if rasp.is_numerical(expr) and not set(agg_in).issubset({0, 1}):
        self.unsupported_exprs.append(
            TracrUnsupportedExpr(
                expr=expr,
                reason=(
                    "Numerical aggregate only supports binary inputs 0, 1. But"
                    f" got {set(agg_in)}."
                ),
            )
        )

    return out


def static_validate(program: rasp.RASPExpr) -> list[TracrUnsupportedExpr]:
  """Performs static checks to see if `program` can be compiled.

  Args:
    program: RASP program to validate

  Returns:
    list of all unsupported subexpressions detectable statically.
  """
  expr_queue = queue.Queue()
  unsupported_exprs = []
  visited_exprs = set()

  # Breadth-first search over the RASP expression graph.
  def visit_raspexpr(expr: rasp.RASPExpr):
    visited_exprs.add(expr.name)
    unsupported_expr = _static_validate_expr(expr)
    if unsupported_expr:
      unsupported_exprs.append(unsupported_expr)

    for child_expr in expr.children:
      if child_expr.name not in visited_exprs:
        expr_queue.put(child_expr)

  expr_queue.put(program)
  while not expr_queue.empty():
    visit_raspexpr(expr_queue.get())

  return unsupported_exprs


def dynamic_validate(
    program: rasp.RASPExpr, xs: Sequence[rasp.Value] | None = None
) -> list[TracrUnsupportedExpr]:
  """Checks if `program` can be compiled for input `xs`.

  Args:
    program: RASP program to validate
    xs: Input sequence to use for dynamic compiler check. If None, only do
      static checks.

  Returns:
    list of all unsupported expressions according to the dynamic validation
  """
  validation_evaluator = DynamicValidationEvaluator()
  validation_evaluator.evaluate(expr=program, xs=xs)
  return validation_evaluator.unsupported_exprs


def validate(
    program: rasp.RASPExpr, xs: Sequence[rasp.Value] | None = None
) -> list[TracrUnsupportedExpr]:
  """Checks if `program` can be compiled for input `xs`.

  Args:
    program: RASP program to validate
    xs: Input sequence to use for dynamic compiler check. If None, only do
      static checks.

  Returns:
    list of all unsupported expressions
  """
  static_unsupported = static_validate(program)
  if xs is not None:
    dynamic_unsupported = dynamic_validate(program, xs)
    return static_unsupported + dynamic_unsupported
  return static_unsupported
