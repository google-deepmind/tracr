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
"""Tests for compiler.lib."""

from absl.testing import absltest
from absl.testing import parameterized
from tracr.compiler import test_cases
from tracr.rasp import causal_eval
from tracr.rasp import rasp


class LibTest(parameterized.TestCase):

  @parameterized.named_parameters(*test_cases.TEST_CASES)
  def test_program_produces_expected_output(self, program, test_input,
                                            expected_output, **kwargs):
    del kwargs
    self.assertEqual(rasp.evaluate(program, test_input), expected_output)

  @parameterized.named_parameters(*test_cases.CAUSAL_TEST_CASES)
  def test_causal_program_produces_expected_output(self, program, test_input,
                                                   expected_output, **kwargs):
    del kwargs
    self.assertEqual(causal_eval.evaluate(program, test_input), expected_output)


if __name__ == "__main__":
  absltest.main()
