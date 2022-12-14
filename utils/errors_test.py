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
"""Tests for rasp.helper."""

from absl.testing import absltest
from absl.testing import parameterized
from tracr.utils import errors


class FunIgnoreArithmeticErrorsTest(parameterized.TestCase):

  def test_ignoring_arithmetic_errors(self):
    fun = lambda x: 1 / x
    fun_ignore = errors.ignoring_arithmetic_errors(fun)

    with self.assertLogs(level="WARNING"):
      res = fun_ignore(0)
    self.assertIs(res, None)

    self.assertEqual(fun_ignore(1), 1)
    self.assertEqual(fun_ignore(2), 0.5)
    self.assertEqual(fun_ignore(-2), -0.5)

  def test_ignoring_arithmetic_errors_two_arguments(self):
    fun = lambda x, y: 1 / x + 1 / y
    fun_ignore = errors.ignoring_arithmetic_errors(fun)

    with self.assertLogs(level="WARNING"):
      res = fun_ignore(0, 1)
    self.assertIs(res, None)

    with self.assertLogs(level="WARNING"):
      res = fun_ignore(0, 0)
    self.assertIs(res, None)

    with self.assertLogs(level="WARNING"):
      res = fun_ignore(1, 0)
    self.assertIs(res, None)

    self.assertEqual(fun_ignore(1, 1), 2)
    self.assertEqual(fun_ignore(1, 2), 1.5)
    self.assertEqual(fun_ignore(-2, 2), 0)


if __name__ == "__main__":
  absltest.main()
