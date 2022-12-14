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
"""Tests for transformers."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tracr.craft import bases
from tracr.craft import tests_common
from tracr.craft import transformers
from tracr.craft import vectorspace_fns as vs_fns

# This makes it easier to use comments to annotate dimensions in arrays
# pylint: disable=g-no-space-after-comment


class AttentionHeadTest(tests_common.VectorFnTestCase):

  @parameterized.parameters([
      dict(with_residual_stream=False),
      dict(with_residual_stream=True),
  ])
  def test_attention_head(self, with_residual_stream):
    i = bases.VectorSpaceWithBasis.from_values("i", [1, 2])
    o = bases.VectorSpaceWithBasis.from_values("o", [1, 2])
    q = bases.VectorSpaceWithBasis.from_values("q", [1, 2])
    k = bases.VectorSpaceWithBasis.from_values("p", [1, 2])
    rs = bases.direct_sum(i, o, q, k)

    seq = bases.VectorInBasis(
        rs.basis,
        np.array([
            #i1 i2 o1 o2 q1 q2 p1 p2
            [1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 1],
        ]))

    head = transformers.AttentionHead(
        w_qk=vs_fns.ScalarBilinear(q, k,
                                   np.eye(2) * 100),
        w_ov=vs_fns.Linear(i, o, np.eye(2)),
        residual_space=rs if with_residual_stream else None,
        causal=False,
    )

    self.assertVectorAllClose(
        head.apply(seq),
        bases.VectorInBasis(
            rs.basis,
            np.array([
                #i1 i2 o1 o2 q1 q2 p1 p2
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ])),
    )


class MLPTest(tests_common.VectorFnTestCase):

  @parameterized.parameters([
      dict(with_residual_stream=False, same_in_out=False),
      dict(with_residual_stream=False, same_in_out=True),
      dict(with_residual_stream=True, same_in_out=False),
      dict(with_residual_stream=True, same_in_out=True),
  ])
  def test_mlp(self, with_residual_stream, same_in_out):
    i = bases.VectorSpaceWithBasis.from_values("i", [1, 2])
    if same_in_out:
      o, rs = i, i
      expected_result = np.array([
          #o1 o2
          [1, 0],
          [0, 1],
      ])
    else:
      o = bases.VectorSpaceWithBasis.from_values("o", [1, 2])
      rs = bases.direct_sum(i, o)
      expected_result = np.array([
          #i1 i2 o1 o2
          [0, 0, 1, 0],
          [0, 0, 0, 1],
      ])
    h = bases.VectorSpaceWithBasis.from_values("p", [1, 2])

    seq = bases.VectorInBasis(
        i.basis,
        np.array([
            #i1  i2
            [1, -1],
            [-1, 1],
        ])).project(rs)

    mlp = transformers.MLP(
        fst=vs_fns.Linear(i, h, np.eye(2)),
        snd=vs_fns.Linear(h, o, np.eye(2)),
        residual_space=rs if with_residual_stream else None,
    )

    self.assertEqual(
        mlp.apply(seq),
        bases.VectorInBasis(rs.basis, expected_result),
    )

  def test_combining_mlps(self):
    in12 = bases.VectorSpaceWithBasis.from_values("in", [1, 2])
    in34 = bases.VectorSpaceWithBasis.from_values("in", [3, 4])
    out12 = bases.VectorSpaceWithBasis.from_values("out", [1, 2])
    residual_space = bases.join_vector_spaces(in12, in34, out12)

    h1 = bases.VectorSpaceWithBasis.from_values("h", [1])
    h2 = bases.VectorSpaceWithBasis.from_values("h", [2])

    # MLP1 maps in2 -> h1 -> out1
    mlp1 = transformers.MLP(
        fst=vs_fns.Linear(in12, h1, np.array([[0], [1]])),
        snd=vs_fns.Linear(h1, out12, np.array([[1, 0]])))

    # MLP2 maps in3 -> h2 -> out2
    mlp2 = transformers.MLP(
        fst=vs_fns.Linear(in34, h2, np.array([[1], [0]])),
        snd=vs_fns.Linear(h2, out12, np.array([[0, 1]])))

    mlp = transformers.MLP.combine_in_parallel([mlp1, mlp2])

    seq = bases.VectorInBasis(
        bases.direct_sum(in12, in34).basis,
        np.array([
            #i1 i2 i3 i4
            [1, 2, 0, 0],
            [0, 2, 3, 4],
        ])).project(residual_space)

    expected_result = bases.VectorInBasis(
        out12.basis,
        np.array([
            #o1 o2
            [2, 0],
            [2, 3],
        ]))

    self.assertEqual(
        mlp.apply(seq).project(out12),
        expected_result,
    )


if __name__ == "__main__":
  absltest.main()
