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
"""Basic encoder for inputs with a fixed vocabulary."""

import abc
from typing import Any, List, Optional, Sequence

from tracr.craft import bases


class Encoder(abc.ABC):
  """Encodes a list of tokens into a list of inputs for a transformer model.

  The abstract class does not make assumptions on the input and output types,
  and we have different encoders for different input types.
  """

  @abc.abstractmethod
  def encode(self, inputs: List[Any]) -> List[Any]:
    return list()

  @abc.abstractmethod
  def decode(self, encodings: List[Any]) -> List[Any]:
    return list()

  @property
  def pad_token(self) -> Optional[str]:
    return None

  @property
  def bos_token(self) -> Optional[str]:
    return None

  @property
  def pad_encoding(self) -> Optional[int]:
    return None

  @property
  def bos_encoding(self) -> Optional[int]:
    return None


class NumericalEncoder(Encoder):
  """Encodes numerical variables (simply using the identity mapping)."""

  def encode(self, inputs: List[float]) -> List[float]:
    return inputs

  def decode(self, encodings: List[float]) -> List[float]:
    return encodings


class CategoricalEncoder(Encoder):
  """Encodes categorical variables with a fixed vocabulary."""

  def __init__(
      self,
      basis: Sequence[bases.BasisDirection],
      enforce_bos: bool = False,
      bos_token: Optional[str] = None,
      pad_token: Optional[str] = None,
      max_seq_len: Optional[int] = None,
  ):
    """Initialises. If enforce_bos is set, ensures inputs start with it."""
    if enforce_bos and not bos_token:
      raise ValueError("BOS token must be specified if enforcing BOS.")

    self.encoding_map = {}
    for i, direction in enumerate(basis):
      val = direction.value
      self.encoding_map[val] = i

    if bos_token and bos_token not in self.encoding_map:
      raise ValueError("BOS token missing in encoding.")

    if pad_token and pad_token not in self.encoding_map:
      raise ValueError("PAD token missing in encoding.")

    self.enforce_bos = enforce_bos
    self._bos_token = bos_token
    self._pad_token = pad_token
    self._max_seq_len = max_seq_len

  def encode(self, inputs: List[bases.Value]) -> List[int]:
    if self.enforce_bos and inputs[0] != self.bos_token:
      raise ValueError("First input token must be BOS token. "
                       f"Should be '{self.bos_token}', but was '{inputs[0]}'.")
    if missing := set(inputs) - set(self.encoding_map.keys()):
      raise ValueError(f"Inputs {missing} not found in encoding ",
                       self.encoding_map.keys())
    if self._max_seq_len is not None and len(inputs) > self._max_seq_len:
      raise ValueError(f"inputs={inputs} are longer than the maximum "
                       f"sequence length {self._max_seq_len}")

    return [self.encoding_map[x] for x in inputs]

  def decode(self, encodings: List[int]) -> List[bases.Value]:
    """Recover the tokens that corresponds to `ids`. Inverse of __call__."""
    decoding_map = {val: key for key, val in self.encoding_map.items()}
    if missing := set(encodings) - set(decoding_map.keys()):
      raise ValueError(f"Inputs {missing} not found in decoding map ",
                       decoding_map.keys())
    return [decoding_map[x] for x in encodings]

  @property
  def vocab_size(self) -> int:
    return len(self.encoding_map)

  @property
  def bos_token(self) -> Optional[str]:
    return self._bos_token

  @property
  def pad_token(self) -> Optional[str]:
    return self._pad_token

  @property
  def bos_encoding(self) -> Optional[int]:
    return None if self.bos_token is None else self.encoding_map[self.bos_token]

  @property
  def pad_encoding(self) -> Optional[int]:
    return None if self.pad_token is None else self.encoding_map[self.pad_token]
