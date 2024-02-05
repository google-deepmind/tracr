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
"""Tests for bases."""

from absl.testing import absltest
import numpy as np
from tracr.craft import bases
from tracr.craft import tests_common


class VectorInBasisTest(tests_common.VectorFnTestCase):

  def test_shape_mismatch_raises_value_error(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    regex = (
        r"^.*Last dimension of magnitudes must be the same as number of "
        r"basis directions.*$"
    )
    with self.assertRaisesRegex(ValueError, regex):
      bases.VectorInBasis(vs1.basis, np.array([1, 2, 3, 4]))
    with self.assertRaisesRegex(ValueError, regex):
      bases.VectorInBasis(vs1.basis, np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))

  def test_equal(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b", "c", "d"])
    v1 = bases.VectorInBasis(vs1.basis, np.array([1, 2, 3, 4]))
    v2 = bases.VectorInBasis(vs1.basis, np.array([1, 2, 3, 4]))
    self.assertEqual(v1, v2)
    self.assertEqual(v2, v1)
    v3 = bases.VectorInBasis(vs1.basis, np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))
    v4 = bases.VectorInBasis(vs1.basis, np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))
    self.assertEqual(v3, v4)
    self.assertEqual(v4, v3)
    v5 = bases.VectorInBasis(vs1.basis, np.array([1, 2, 3, 4]))
    v6 = bases.VectorInBasis(vs1.basis, np.array([1, 1, 1, 1]))
    self.assertNotEqual(v5, v6)
    self.assertNotEqual(v6, v5)
    v7 = bases.VectorInBasis(vs1.basis, np.array([1, 2, 3, 4]))
    v8 = bases.VectorInBasis(vs1.basis, np.array([[1, 2, 3, 4], [1, 1, 1, 1]]))
    self.assertNotEqual(v7, v8)
    self.assertNotEqual(v8, v7)
    vs2 = bases.VectorSpaceWithBasis.from_names(["e", "f", "g", "h"])
    v9 = bases.VectorInBasis(vs1.basis, np.array([1, 2, 3, 4]))
    v10 = bases.VectorInBasis(vs2.basis, np.array([1, 2, 3, 4]))
    self.assertNotEqual(v9, v10)
    self.assertNotEqual(v10, v9)

  def test_dunders(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b", "c"])
    v = bases.VectorInBasis(vs1.basis, np.array([0, 1, 2]))
    three = bases.VectorInBasis(vs1.basis, np.array([3, 3, 3]))
    five = bases.VectorInBasis(vs1.basis, np.array([5, 5, 5]))
    v_times_5 = bases.VectorInBasis(vs1.basis, np.array([0, 5, 10]))
    self.assertEqual(5 * v, v_times_5)
    self.assertEqual(v * 5, v_times_5)
    self.assertEqual(5.0 * v, v_times_5)
    self.assertEqual(v * 5.0, v_times_5)
    v_by_2 = bases.VectorInBasis(vs1.basis, np.array([0, 0.5, 1]))
    self.assertEqual(v / 2, v_by_2)
    self.assertEqual(v / 2.0, v_by_2)
    self.assertEqual(1 / 2 * v, v_by_2)
    v_plus_3 = bases.VectorInBasis(vs1.basis, np.array([3, 4, 5]))
    self.assertEqual(v + three, v_plus_3)
    self.assertEqual(three + v, v_plus_3)
    v_minus_5 = bases.VectorInBasis(vs1.basis, np.array([-5, -4, -3]))
    self.assertEqual(v - five, v_minus_5)
    minus_v = bases.VectorInBasis(vs1.basis, np.array([0, -1, -2]))
    self.assertEqual(-v, minus_v)

  def test_add_directions(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b", "c"])
    expected = bases.VectorInBasis(vs1.basis, np.array([3, 4, 5]))
    v = bases.VectorInBasis(vs1.basis, np.array([0, 1, 2]))
    three = bases.VectorInBasis(vs1.basis, np.array([3, 3, 3]))
    shifted = v.add_directions(three)
    self.assertEqual(shifted, expected)


class ProjectionTest(tests_common.VectorFnTestCase):

  def test_direct_sum_produces_expected_result(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    vs2 = bases.VectorSpaceWithBasis.from_names(["d", "c"])
    vs3 = bases.VectorSpaceWithBasis.from_names(["a", "b", "d", "c"])
    self.assertEqual(bases.direct_sum(vs1, vs2), vs3)

  def test_join_vector_spaces_produces_expected_result(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    vs2 = bases.VectorSpaceWithBasis.from_names(["d", "c"])
    vs3 = bases.VectorSpaceWithBasis.from_names(["a", "b", "c", "d"])
    self.assertEqual(bases.join_vector_spaces(vs1, vs2), vs3)

    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    vs2 = bases.VectorSpaceWithBasis.from_names(["b", "d", "c"])
    vs3 = bases.VectorSpaceWithBasis.from_names(["a", "b", "c", "d"])
    self.assertEqual(bases.join_vector_spaces(vs1, vs2), vs3)

  def test_compare_vectors_with_differently_ordered_basis_vectors(self):
    basis1 = ["a", "b", "c", "d"]
    basis1 = [bases.BasisDirection(x) for x in basis1]
    basis2 = ["b", "d", "a", "c"]
    basis2 = [bases.BasisDirection(x) for x in basis2]
    vs1 = bases.VectorSpaceWithBasis(basis1)
    vs2 = bases.VectorSpaceWithBasis(basis2)
    v1 = bases.VectorInBasis(basis1, np.array([1, 2, 3, 4]))
    v2 = bases.VectorInBasis(basis2, np.array([2, 4, 1, 3]))
    self.assertEqual(v1, v2)
    self.assertEqual(v1 - v2, vs1.null_vector())
    self.assertEqual(v1 - v2, vs2.null_vector())
    self.assertEqual(v1 + v2, 2 * v2)
    self.assertIn(v1, vs1)
    self.assertIn(v1, vs2)
    self.assertIn(v2, vs1)
    self.assertIn(v2, vs2)

  def test_compare_vector_arrays_with_differently_ordered_basis_vectors(self):
    basis1 = ["a", "b", "c", "d"]
    basis1 = [bases.BasisDirection(x) for x in basis1]
    basis2 = ["b", "d", "a", "c"]
    basis2 = [bases.BasisDirection(x) for x in basis2]
    vs1 = bases.VectorSpaceWithBasis(basis1)
    vs2 = bases.VectorSpaceWithBasis(basis2)
    v1 = bases.VectorInBasis(basis1, np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
    v2 = bases.VectorInBasis(basis2, np.array([[2, 4, 1, 3], [6, 8, 5, 7]]))
    null_vec = bases.VectorInBasis.stack([vs1.null_vector(), vs2.null_vector()])
    self.assertEqual(v1, v2)
    self.assertEqual(v1 - v2, null_vec)
    self.assertEqual(v1 + v2, 2 * v2)
    self.assertIn(v1, vs1)
    self.assertIn(v1, vs2)
    self.assertIn(v2, vs1)
    self.assertIn(v2, vs2)

  def test_projection_to_larger_space(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    vs2 = bases.VectorSpaceWithBasis.from_names(["a", "b", "c", "d"])
    a1, b1 = vs1.basis_vectors()
    a2, b2, _, _ = vs2.basis_vectors()

    self.assertEqual(a1.project(vs2), a2)
    self.assertEqual(b1.project(vs2), b2)

  def test_projection_to_smaller_space(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b", "c", "d"])
    vs2 = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    a1, b1, c1, d1 = vs1.basis_vectors()
    a2, b2 = vs2.basis_vectors()

    self.assertEqual(a1.project(vs2), a2)
    self.assertEqual(b1.project(vs2), b2)
    self.assertEqual(c1.project(vs2), vs2.null_vector())
    self.assertEqual(d1.project(vs2), vs2.null_vector())


if __name__ == "__main__":
  absltest.main()
