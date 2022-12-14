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
"""Tests for causal_eval."""

from absl.testing import absltest
from absl.testing import parameterized

from tracr.rasp import causal_eval
from tracr.rasp import rasp


class CausalEvalTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="constant_selector_3x3_1",
          program=rasp.ConstantSelector([
              [True, True, True],
              [True, True, True],
              [True, True, True],
          ]),
          input_sequence=[True, True, True],
          expected_output=[
              [True, False, False],
              [True, True, False],
              [True, True, True],
          ]),
      dict(
          testcase_name="constant_selector_3x3_2",
          program=rasp.ConstantSelector([
              [True, True, True],
              [False, True, True],
              [True, False, True],
          ]),
          input_sequence=[True, True, True],
          expected_output=[
              [True, False, False],
              [False, True, False],
              [True, False, True],
          ]))
  def test_evaluations(self, program, input_sequence, expected_output):
    self.assertListEqual(
        causal_eval.evaluate(program, input_sequence),
        expected_output,
    )


if __name__ == "__main__":
  absltest.main()
