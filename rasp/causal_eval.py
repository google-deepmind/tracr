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

from typing import Sequence, Union

import numpy as np
from tracr.rasp import rasp


class CausalEvaluator(rasp.DefaultRASPEvaluator):
  """Evaluates RASP with causal masking."""

  def evaluate(
      self, expr: rasp.RASPExpr, xs: Sequence[rasp.Value]
  ) -> Union[Sequence[rasp.Value], rasp.SelectorValue]:
    out = super().evaluate(expr, xs)

    if not isinstance(expr, rasp.Selector):
      return out

    out = np.array(out)
    causal_mask = np.tril(np.full(out.shape, 1))
    return np.logical_and(causal_mask, out).tolist()


evaluate = CausalEvaluator().evaluate
