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
"""Tests for compiler.compilable_evaluator."""

from absl.testing import absltest
from absl.testing import parameterized
from tracr.compiler import test_cases
from tracr.compiler import validating
from tracr.rasp import rasp


class ValidationEvaluatorTest(parameterized.TestCase):

  @parameterized.named_parameters(test_cases.TEST_CASES)
  def test_supported_programs_pass_validation(
      self,
      program,
      test_input,
      **kwargs,
  ):
    del kwargs
    validation_result = validating.validate(program, test_input)
    self.assertEmpty(validation_result)

  @parameterized.named_parameters(test_cases.UNSUPPORTED_TEST_CASES)
  def test_unsupported_programs_fail_validation(
      self,
      program,
      vocab,
      **kwargs,
  ):
    del kwargs
    test_input = sorted(list(vocab))
    validation_result = validating.validate(program, test_input)
    self.assertNotEmpty(validation_result)

  @parameterized.named_parameters(
      dict(
          testcase_name="mean",
          program=rasp.Aggregate(
              rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE),
              rasp.tokens,
          ),
          test_input=[1, 2, 3, 4],
      ),
      dict(
          testcase_name="prev_mean",
          program=rasp.Aggregate(
              rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.LEQ),
              rasp.tokens,
          ),
          test_input=[1, 2, 3, 4],
      ),
  )
  def test_dynamic_failure_cases_fail_validation(
      self,
      program,
      test_input,
  ):
    # Dynamic test cases are not in the general test case suite because they are
    # not caught at compile time.
    validation_result = validating.validate(program, test_input)
    self.assertNotEmpty(validation_result)


if __name__ == "__main__":
  absltest.main()
