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
"""Tests for rasp.rasp."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tracr.rasp import rasp

# Note that the example text labels must match their default names.

_SOP_PRIMITIVE_EXAMPLES = lambda: [  # pylint: disable=g-long-lambda
    ("tokens", rasp.tokens),
    ("length", rasp.length),
    ("indices", rasp.indices),
]

_NONPRIMITIVE_SOP_EXAMPLES = lambda: [  # pylint: disable=g-long-lambda
    ("map", rasp.Map(lambda x: x, rasp.tokens)),
    (
        "sequence_map",
        rasp.SequenceMap(lambda x, y: x + y, rasp.tokens, rasp.tokens),
    ),
    (
        "linear_sequence_map",
        rasp.LinearSequenceMap(rasp.tokens, rasp.tokens, 0.1, 0.2),
    ),
    (
        "aggregate",
        rasp.Aggregate(
            rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ),
            rasp.tokens,
        ),
    ),
    (
        "selector_width",
        rasp.SelectorWidth(
            rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ)),
    ),
]

_SOP_EXAMPLES = lambda: _SOP_PRIMITIVE_EXAMPLES() + _NONPRIMITIVE_SOP_EXAMPLES()

_SELECTOR_EXAMPLES = lambda: [  # pylint: disable=g-long-lambda
    ("select", rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ)),
    ("selector_and",
     rasp.SelectorAnd(
         rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ),
         rasp.Select(rasp.indices, rasp.tokens, rasp.Comparison.LEQ),
     )),
    ("selector_or",
     rasp.SelectorOr(
         rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ),
         rasp.Select(rasp.indices, rasp.tokens, rasp.Comparison.LEQ),
     )),
    ("selector_not",
     rasp.SelectorNot(
         rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ),)),
]

_ALL_EXAMPLES = lambda: _SOP_EXAMPLES() + _SELECTOR_EXAMPLES()


class LabelTest(parameterized.TestCase):

  def test_primitive_labels(self):
    self.assertEqual(rasp.tokens.label, "tokens")
    self.assertEqual(rasp.indices.label, "indices")
    self.assertEqual(rasp.length.label, "length")

  @parameterized.parameters(*_ALL_EXAMPLES())
  def test_default_names(self, default_name: str, expr: rasp.RASPExpr):
    self.assertEqual(expr.name, default_name)


class SOpTest(parameterized.TestCase):
  """Tests for S-Ops."""

  @parameterized.parameters(
      ("hello", ["h", "e", "l", "l", "o"]),
      ("h", ["h"]),
      (["h", "e", "l", "l", "o"], ["h", "e", "l", "l", "o"]),
      (["h"], ["h"]),
      ([1, 2], [1, 2]),
      ([0.1, 0.2], [0.1, 0.2]),
  )
  def test_tokens(self, input_sequence, expected):
    self.assertEqual(rasp.tokens(input_sequence), expected)

  @parameterized.parameters(
      ("hello", [0, 1, 2, 3, 4]),
      ("h", [0]),
      (["h", "e", "l", "l", "o"], [0, 1, 2, 3, 4]),
      (["h"], [0]),
      ([1, 2], [0, 1]),
      ([0.1, 0.2], [0, 1]),
  )
  def test_indices(self, input_sequence, expected):
    self.assertEqual(rasp.indices(input_sequence), expected)

  @parameterized.parameters(
      ("hello", [5, 5, 5, 5, 5]),
      ("h", [1]),
      (["h", "e", "l", "l", "o"], [5, 5, 5, 5, 5]),
      (["h"], [1]),
      ([1, 2], [2, 2]),
      ([0.1, 0.2], [2, 2]),
  )
  def test_length(self, input_sequence, expected):
    self.assertEqual(rasp.length(input_sequence), expected)

  def test_prims_are_sops(self):
    self.assertIsInstance(rasp.tokens, rasp.SOp)
    self.assertIsInstance(rasp.indices, rasp.SOp)
    self.assertIsInstance(rasp.length, rasp.SOp)

  def test_prims_are_raspexprs(self):
    self.assertIsInstance(rasp.tokens, rasp.RASPExpr)
    self.assertIsInstance(rasp.indices, rasp.RASPExpr)
    self.assertIsInstance(rasp.length, rasp.RASPExpr)

  @parameterized.parameters(
      (lambda x: x + "a", "hello", ["ha", "ea", "la", "la", "oa"]),
      (lambda x: x + "t", "h", ["ht"]),
      (lambda x: x + 1, [1, 2], [2, 3]),
      (lambda x: x / 2, [0.1, 0.2], [0.05, 0.1]),
  )
  def test_map(self, f, input_sequence, expected):
    self.assertEqual(rasp.Map(f, rasp.tokens)(input_sequence), expected)

  def test_nested_elementwise_ops_results_in_only_one_map_object(self):
    map_sop = ((rasp.tokens * 2) + 2) / 2
    self.assertEqual(map_sop.inner, rasp.tokens)
    self.assertEqual(map_sop([1]), [2])

  def test_nested_maps_result_in_two_map_objects_if_simplify_set_to_false(self):
    map_sop = rasp.Map(lambda x: x + 2, (rasp.tokens * 2), simplify=False)
    self.assertNotEqual(map_sop.inner, rasp.tokens)
    self.assertEqual(map_sop([1]), [4])

  @parameterized.parameters(
      (lambda x, y: x + y, "hello", ["hh", "ee", "ll", "ll", "oo"]),
      (lambda x, y: x + y, "h", ["hh"]),
      (lambda x, y: x + y, [1, 2], [2, 4]),
      (lambda x, y: x * y, [1, 2], [1, 4]),
  )
  def test_sequence_map(self, f, input_sequence, expected):
    self.assertEqual(
        rasp.SequenceMap(f, rasp.tokens, rasp.tokens)(input_sequence), expected)

  def test_sequence_map_with_same_inputs_logs_warning(self):
    with self.assertLogs(level="WARNING"):
      rasp.SequenceMap(lambda x, y: x + y, rasp.tokens, rasp.tokens)

  @parameterized.parameters(
      (1, 1, [1, 2], [2, 4]),
      (1, -1, [1, 2], [0, 0]),
      (1, -2, [1, 2], [-1, -2]),
  )
  def test_linear_sequence_map(self, fst_fac, snd_fac, input_sequence,
                               expected):
    self.assertEqual(
        rasp.LinearSequenceMap(rasp.tokens, rasp.tokens, fst_fac,
                               snd_fac)(input_sequence), expected)

  @parameterized.parameters(
      ([5, 5, 5, 5, 5], "hello", [5, 5, 5, 5, 5]),
      (["e"], "h", ["e"]),
      ([1, 2, 3, 4, 5], ["h", "e", "l", "l", "o"], [1, 2, 3, 4, 5]),
      ([2, 2], [1, 2], [2, 2]),
  )
  def test_constant(self, const, input_sequence, expected):
    self.assertEqual(rasp.ConstantSOp(const)(input_sequence), expected)

  def test_constant_complains_if_sizes_dont_match(self):
    with self.assertRaisesRegex(
        ValueError,
        r"^.*Constant len .* doesn't match input len .*$",):
      rasp.ConstantSOp([1, 2, 3])("longer string")

  def test_can_turn_off_constant_complaints(self):
    rasp.ConstantSOp([1, 2, 3], check_length=False)("longer string")

  def test_numeric_dunders(self):
    # We don't check all the cases here -- only a few representative ones.
    self.assertEqual(
        (rasp.tokens > 1)([0, 1, 2]),
        [0, 0, 1],
    )
    self.assertEqual(
        (1 < rasp.tokens)([0, 1, 2]),
        [0, 0, 1],
    )
    self.assertEqual(
        (rasp.tokens < 1)([0, 1, 2]),
        [1, 0, 0],
    )
    self.assertEqual(
        (1 > rasp.tokens)([0, 1, 2]),
        [1, 0, 0],
    )
    self.assertEqual(
        (rasp.tokens == 1)([0, 1, 2]),
        [0, 1, 0],
    )
    self.assertEqual(
        (rasp.tokens + 1)([0, 1, 2]),
        [1, 2, 3],
    )
    self.assertEqual(
        (1 + rasp.tokens)([0, 1, 2]),
        [1, 2, 3],
    )

  def test_dunders_with_sop(self):
    self.assertEqual(
        (rasp.tokens + rasp.indices)([0, 1, 2]),
        [0, 2, 4],
    )
    self.assertEqual(
        (rasp.length - 1 - rasp.indices)([0, 1, 2]),
        [2, 1, 0],
    )
    self.assertEqual(
        (rasp.length * rasp.length)([0, 1, 2]),
        [9, 9, 9],
    )

  def test_logical_dunders(self):
    self.assertEqual(
        (rasp.tokens & True)([True, False]),
        [True, False],
    )
    self.assertEqual(
        (rasp.tokens & False)([True, False]),
        [False, False],
    )
    self.assertEqual(
        (rasp.tokens | True)([True, False]),
        [True, True],
    )
    self.assertEqual(
        (rasp.tokens | False)([True, False]),
        [True, False],
    )
    self.assertEqual(
        (True & rasp.tokens)([True, False]),
        [True, False],
    )
    self.assertEqual(
        (False & rasp.tokens)([True, False]),
        [False, False],
    )
    self.assertEqual(
        (True | rasp.tokens)([True, False]),
        [True, True],
    )
    self.assertEqual(
        (False | rasp.tokens)([True, False]),
        [True, False],
    )

    self.assertEqual(
        (~rasp.tokens)([True, False]),
        [False, True],
    )

    self.assertEqual(
        (rasp.ConstantSOp([True, True, False, False])
         & rasp.ConstantSOp([True, False, True, False]))([1, 1, 1, 1]),
        [True, False, False, False],
    )

    self.assertEqual(
        (rasp.ConstantSOp([True, True, False, False])
         | rasp.ConstantSOp([True, False, True, False]))([1, 1, 1, 1]),
        [True, True, True, False],
    )


class EncodingTest(parameterized.TestCase):
  """Tests for SOp encodings."""

  @parameterized.named_parameters(*_SOP_EXAMPLES())
  def test_all_sops_are_categorical_by_default(self, sop: rasp.SOp):
    self.assertTrue(rasp.is_categorical(sop))

  @parameterized.named_parameters(*_SOP_EXAMPLES())
  def test_is_numerical(self, sop: rasp.SOp):
    self.assertTrue(rasp.is_numerical(rasp.numerical(sop)))
    self.assertFalse(rasp.is_numerical(rasp.categorical(sop)))

  @parameterized.named_parameters(*_SOP_EXAMPLES())
  def test_is_categorical(self, sop: rasp.SOp):
    self.assertTrue(rasp.is_categorical(rasp.categorical(sop)))
    self.assertFalse(rasp.is_categorical(rasp.numerical(sop)))

  @parameterized.named_parameters(*_SOP_EXAMPLES())
  def test_double_encoding_annotations_overwrites_encoding(self, sop: rasp.SOp):
    num_sop = rasp.numerical(sop)
    cat_num_sop = rasp.categorical(num_sop)
    self.assertTrue(rasp.is_numerical(num_sop))
    self.assertTrue(rasp.is_categorical(cat_num_sop))


class SelectorTest(parameterized.TestCase):
  """Tests for Selectors."""

  def test_select_eq_has_correct_value(self):
    selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ)
    self.assertEqual(
        selector("hey"), [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ])

  def test_select_lt_has_correct_value(self):
    selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.LT)
    self.assertEqual(selector([0, 1]), [
        [False, False],
        [True, False],
    ])

  def test_select_leq_has_correct_value(self):
    selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.LEQ)
    self.assertEqual(selector([0, 1]), [
        [True, False],
        [True, True],
    ])

  def test_select_gt_has_correct_value(self):
    selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.GT)
    self.assertEqual(selector([0, 1]), [
        [False, True],
        [False, False],
    ])

  def test_select_geq_has_correct_value(self):
    selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.GEQ)
    self.assertEqual(selector([0, 1]), [
        [True, True],
        [False, True],
    ])

  def test_select_neq_has_correct_value(self):
    selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.NEQ)
    self.assertEqual(selector([0, 1]), [
        [False, True],
        [True, False],
    ])

  def test_select_true_has_correct_value(self):
    selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
    self.assertEqual(selector([0, 1]), [
        [True, True],
        [True, True],
    ])

  def test_select_false_has_correct_value(self):
    selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.FALSE)
    self.assertEqual(selector([0, 1]), [
        [False, False],
        [False, False],
    ])

  def test_selector_and_gets_simplified_when_keys_and_queries_match(self):
    selector = rasp.selector_and(
        rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.GEQ),
        rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.LEQ),
    )
    self.assertIsInstance(selector, rasp.Select)
    self.assertIs(selector.keys, rasp.tokens)
    self.assertIs(selector.queries, rasp.indices)

  def test_selector_and_doesnt_get_simplified_when_keys_queries_different(self):
    selector = rasp.selector_and(
        rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.GEQ),
        rasp.Select(rasp.indices, rasp.tokens, rasp.Comparison.LEQ),
    )
    self.assertIsInstance(selector, rasp.SelectorAnd)

  def test_selector_and_gets_simplified_when_keys_are_full(self):
    selector = rasp.selector_and(
        rasp.Select(rasp.Full(1), rasp.indices, rasp.Comparison.GEQ),
        rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.LEQ),
    )
    self.assertIsInstance(selector, rasp.Select)
    self.assertIs(selector.keys, rasp.tokens)
    self.assertIs(selector.queries, rasp.indices)

  def test_selector_and_gets_simplified_when_queries_are_full(self):
    selector = rasp.selector_and(
        rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.GEQ),
        rasp.Select(rasp.tokens, rasp.Full(1), rasp.Comparison.LEQ),
    )
    self.assertIsInstance(selector, rasp.Select)
    self.assertIs(selector.keys, rasp.tokens)
    self.assertIs(selector.queries, rasp.indices)

  @parameterized.parameters(
      itertools.product(
          (rasp.tokens, rasp.indices, rasp.Full(1)),
          (rasp.tokens, rasp.indices, rasp.Full(1)),
          list(rasp.Comparison),
          (rasp.tokens, rasp.indices, rasp.Full(1)),
          (rasp.tokens, rasp.indices, rasp.Full(1)),
          list(rasp.Comparison),
      ))
  def test_simplified_selector_and_works_the_same_way_as_not(
      self, fst_k, fst_q, fst_p, snd_k, snd_q, snd_p):
    fst = rasp.Select(fst_k, fst_q, fst_p)
    snd = rasp.Select(snd_k, snd_q, snd_p)

    simplified = rasp.selector_and(fst, snd)([0, 1, 2, 3])
    not_simplified = rasp.selector_and(fst, snd, simplify=False)([0, 1, 2, 3])

    np.testing.assert_array_equal(
        np.array(simplified),
        np.array(not_simplified),
    )

  def test_select_is_selector(self):
    self.assertIsInstance(
        rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ),
        rasp.Selector,
    )

  def test_select_is_raspexpr(self):
    self.assertIsInstance(
        rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ),
        rasp.RASPExpr,
    )

  def test_constant_selector(self):
    self.assertEqual(
        rasp.ConstantSelector([[True, True], [False, False]])([1, 2]),
        [[True, True], [False, False]],
    )


class CopyTest(parameterized.TestCase):

  @parameterized.named_parameters(*_ALL_EXAMPLES())
  def test_copy_preserves_name(self, expr: rasp.RASPExpr):
    expr = expr.named("foo")
    self.assertEqual(expr.copy().name, expr.name)

  @parameterized.named_parameters(*_ALL_EXAMPLES())
  def test_renaming_copy_doesnt_rename_original(self, expr: rasp.RASPExpr):
    expr = expr.named("foo")
    expr.copy().named("bar")
    self.assertEqual(expr.name, "foo")

  @parameterized.named_parameters(*_ALL_EXAMPLES())
  def test_renaming_original_doesnt_rename_copy(self, expr: rasp.RASPExpr):
    expr = expr.named("foo")
    copy = expr.copy()
    expr.named("bar")
    self.assertEqual(copy.name, "foo")

  @parameterized.named_parameters(*_ALL_EXAMPLES())
  def test_copy_changes_id(self, expr: rasp.RASPExpr):
    self.assertNotEqual(expr.copy().unique_id, expr.unique_id)

  @parameterized.named_parameters(*_ALL_EXAMPLES())
  def test_copy_preserves_child_ids(self, expr: rasp.RASPExpr):
    copy_child_ids = [c.unique_id for c in expr.copy().children]
    child_ids = [c.unique_id for c in expr.children]
    for child_id, copy_child_id in zip(child_ids, copy_child_ids):
      self.assertEqual(child_id, copy_child_id)


class AggregateTest(parameterized.TestCase):
  """Tests for Aggregate."""

  @parameterized.parameters(
      dict(
          selector=rasp.ConstantSelector([
              [True, False],
              [False, True],
          ]),
          sop=rasp.ConstantSOp(["h", "e"]),
          default=None,
          expected_value=["h", "e"],
      ),
      dict(
          selector=rasp.ConstantSelector([
              [False, True],
              [False, False],
          ]),
          sop=rasp.ConstantSOp(["h", "e"]),
          default=None,
          expected_value=["e", None],
      ),
      dict(
          selector=rasp.ConstantSelector([
              [True, False],
              [False, False],
          ]),
          sop=rasp.ConstantSOp(["h", "e"]),
          default=None,
          expected_value=["h", None],
      ),
      dict(
          selector=rasp.ConstantSelector([
              [True, True],
              [False, True],
          ]),
          sop=rasp.ConstantSOp([0, 1]),
          default=0,
          expected_value=[0.5, 1],
      ),
      dict(
          selector=rasp.ConstantSelector([
              [False, False],
              [True, True],
          ]),
          sop=rasp.ConstantSOp([0, 1]),
          default=0,
          expected_value=[0, 0.5],
      ),
      dict(
          selector=rasp.ConstantSelector([
              [False, False],
              [True, True],
          ]),
          sop=rasp.ConstantSOp([0, 1]),
          default=None,
          expected_value=[None, 0.5],
      ),
  )
  def test_aggregate_on_size_2_inputs(self, selector, sop, default,
                                      expected_value):
    # The 0, 0 input is ignored as it's overridden by the constant SOps.
    self.assertEqual(
        rasp.Aggregate(selector, sop, default)([0, 0]),
        expected_value,
    )


class RaspProgramTest(parameterized.TestCase):
  """Each testcase implements and tests a RASP program."""

  def test_has_prev(self):

    def has_prev(seq: rasp.SOp) -> rasp.SOp:
      prev_copy = rasp.SelectorAnd(
          rasp.Select(seq, seq, rasp.Comparison.EQ),
          rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LT),
      )
      return rasp.Aggregate(prev_copy, rasp.Full(1), default=0) > 0

    self.assertEqual(
        has_prev(rasp.tokens)("hello"),
        [0, 0, 0, 1, 0],
    )

    self.assertEqual(
        has_prev(rasp.tokens)("helllo"),
        [0, 0, 0, 1, 1, 0],
    )

    self.assertEqual(
        has_prev(rasp.tokens)([0, 2, 3, 2, 1, 0, 2]),
        [0, 0, 0, 1, 0, 1, 1],
    )


if __name__ == "__main__":
  absltest.main()
