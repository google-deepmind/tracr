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
"""Tests for vectorspace_fns."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tracr.craft import bases
from tracr.craft import tests_common
from tracr.craft import vectorspace_fns as vs_fns


class LinearTest(tests_common.VectorFnTestCase):

  def test_identity_from_matrix(self):
    vs = bases.VectorSpaceWithBasis.from_names(["a", "b", "c"])
    f = vs_fns.Linear(vs, vs, np.eye(3))
    for v in vs.basis_vectors():
      self.assertEqual(f(v), v)

  def test_identity_from_action(self):
    vs = bases.VectorSpaceWithBasis.from_names(["a", "b", "c"])
    f = vs_fns.Linear.from_action(vs, vs, vs.vector_from_basis_direction)
    for v in vs.basis_vectors():
      self.assertEqual(f(v), v)

  def test_nonidentiy(self):
    vs = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    a = vs.vector_from_basis_direction(bases.BasisDirection("a"))
    b = vs.vector_from_basis_direction(bases.BasisDirection("b"))

    f = vs_fns.Linear(vs, vs, np.array([[0.3, 0.7], [0.2, 0.1]]))

    self.assertEqual(
        f(a), bases.VectorInBasis(vs.basis, np.array([0.3, 0.7])))
    self.assertEqual(
        f(b), bases.VectorInBasis(vs.basis, np.array([0.2, 0.1])))

  def test_different_vector_spaces(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    vs2 = bases.VectorSpaceWithBasis.from_names(["c", "d"])
    a, b = vs1.basis_vectors()
    c, d = vs2.basis_vectors()

    f = vs_fns.Linear(vs1, vs2, np.eye(2))

    self.assertEqual(f(a), c)
    self.assertEqual(f(b), d)

  def test_combining_linear_functions_with_different_input(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    vs2 = bases.VectorSpaceWithBasis.from_names(["c", "d"])
    vs = bases.direct_sum(vs1, vs2)
    a = vs.vector_from_basis_direction(bases.BasisDirection("a"))
    b = vs.vector_from_basis_direction(bases.BasisDirection("b"))
    c = vs.vector_from_basis_direction(bases.BasisDirection("c"))
    d = vs.vector_from_basis_direction(bases.BasisDirection("d"))

    f1 = vs_fns.Linear(vs1, vs1, np.array([[0, 1], [1, 0]]))
    f2 = vs_fns.Linear(vs2, vs2, np.array([[1, 0], [0, 0]]))
    f3 = vs_fns.Linear.combine_in_parallel([f1, f2])

    self.assertEqual(
        f3(a), bases.VectorInBasis(vs.basis, np.array([0, 1, 0, 0])))
    self.assertEqual(
        f3(b), bases.VectorInBasis(vs.basis, np.array([1, 0, 0, 0])))
    self.assertEqual(
        f3(c), bases.VectorInBasis(vs.basis, np.array([0, 0, 1, 0])))
    self.assertEqual(
        f3(d), bases.VectorInBasis(vs.basis, np.array([0, 0, 0, 0])))

  def test_combining_linear_functions_with_same_input(self):
    vs = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    a = vs.vector_from_basis_direction(bases.BasisDirection("a"))
    b = vs.vector_from_basis_direction(bases.BasisDirection("b"))

    f1 = vs_fns.Linear(vs, vs, np.array([[0, 1], [1, 0]]))
    f2 = vs_fns.Linear(vs, vs, np.array([[1, 0], [0, 0]]))
    f3 = vs_fns.Linear.combine_in_parallel([f1, f2])

    self.assertEqual(
        f3(a), bases.VectorInBasis(vs.basis, np.array([1, 1])))
    self.assertEqual(
        f3(b), bases.VectorInBasis(vs.basis, np.array([1, 0])))
    self.assertEqual(f3(a), f1(a) + f2(a))
    self.assertEqual(f3(b), f1(b) + f2(b))


class ProjectionTest(tests_common.VectorFnTestCase):

  def test_projection_to_larger_space(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    vs2 = bases.VectorSpaceWithBasis.from_names(["a", "b", "c", "d"])
    a1, b1 = vs1.basis_vectors()
    a2, b2, _, _ = vs2.basis_vectors()

    f = vs_fns.project(vs1, vs2)

    self.assertEqual(f(a1), a2)
    self.assertEqual(f(b1), b2)

  def test_projection_to_smaller_space(self):
    vs1 = bases.VectorSpaceWithBasis.from_names(["a", "b", "c", "d"])
    vs2 = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    a1, b1, c1, d1 = vs1.basis_vectors()
    a2, b2 = vs2.basis_vectors()

    f = vs_fns.project(vs1, vs2)

    self.assertEqual(f(a1), a2)
    self.assertEqual(f(b1), b2)
    self.assertEqual(f(c1), vs2.null_vector())
    self.assertEqual(f(d1), vs2.null_vector())


class ScalarBilinearTest(parameterized.TestCase):

  def test_identity_matrix(self):
    vs = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    a, b = vs.basis_vectors()

    f = vs_fns.ScalarBilinear(vs, vs, np.eye(2))

    self.assertEqual(f(a, a), 1)
    self.assertEqual(f(a, b), 0)
    self.assertEqual(f(b, a), 0)
    self.assertEqual(f(b, b), 1)

  def test_identity_from_action(self):
    vs = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    a, b = vs.basis_vectors()

    f = vs_fns.ScalarBilinear.from_action(vs, vs, lambda x, y: int(x == y))

    self.assertEqual(f(a, a), 1)
    self.assertEqual(f(a, b), 0)
    self.assertEqual(f(b, a), 0)
    self.assertEqual(f(b, b), 1)

  def test_non_identity(self):
    vs = bases.VectorSpaceWithBasis.from_names(["a", "b"])
    a, b = vs.basis_vectors()

    f = vs_fns.ScalarBilinear.from_action(vs, vs,
                                          lambda x, y: int(x.name == "a"))

    self.assertEqual(f(a, a), 1)
    self.assertEqual(f(a, b), 1)
    self.assertEqual(f(b, a), 0)
    self.assertEqual(f(b, b), 0)


if __name__ == "__main__":
  absltest.main()
