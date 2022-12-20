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
"""Useful helpers for model debugging."""


def print_arrays(arrays, labels=None, colwidth=12):
  """Pretty-prints a list of [1, T, D] arrays."""
  if labels is not None:
    print(" |".join(labels))
    widths = [len(l) for l in labels]
  else:
    widths = [colwidth] * len(arrays[0].shape[-1])
  for layer in arrays:
    print("=" * (colwidth + 1) * layer.shape[1])
    for row in layer[0]:
      print(" |".join([f"{x:<{width}.2f}" for x, width in zip(row, widths)]))
