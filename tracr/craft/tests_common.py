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
"""Helper functions for tests."""

from absl.testing import parameterized
import numpy as np
from tracr.craft import bases


def strip_bos_token(vector: bases.VectorInBasis) -> bases.VectorInBasis:
  """Removes BOS token of a vector."""
  return bases.VectorInBasis(vector.basis_directions, vector.magnitudes[1:])


class VectorFnTestCase(parameterized.TestCase):
  """Asserts for vectors."""

  def assertVectorAllClose(self, v1: bases.VectorInBasis,
                           v2: bases.VectorInBasis):
    self.assertEqual(v1.basis_directions, v2.basis_directions)
    np.testing.assert_allclose(v1.magnitudes, v2.magnitudes, atol=1e-7)
