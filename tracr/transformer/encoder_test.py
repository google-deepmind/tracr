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
"""Tests for transformer.encoder."""

from absl.testing import absltest
from absl.testing import parameterized
from tracr.craft import bases
from tracr.transformer import encoder

_BOS_TOKEN = "bos_encoder_test"
_PAD_TOKEN = "pad_encoder_test"


class CategoricalEncoderTest(parameterized.TestCase):

  def test_encode_raises_value_error_if_input_doesnt_start_with_bos(self):
    vs = bases.VectorSpaceWithBasis.from_values("input", {1, 2, 3, _BOS_TOKEN})
    basic_encoder = encoder.CategoricalEncoder(
        vs.basis, enforce_bos=True, bos_token=_BOS_TOKEN)
    with self.assertRaisesRegex(ValueError,
                                r"^.*First input token must be BOS token.*$"):
      basic_encoder.encode([1, 1, 1])

  def test_encode_raises_value_error_if_input_not_in_vocab(self):
    vs = bases.VectorSpaceWithBasis.from_values("input", {1, 2, 3, _BOS_TOKEN})
    basic_encoder = encoder.CategoricalEncoder(
        vs.basis, enforce_bos=True, bos_token=_BOS_TOKEN)
    with self.assertRaisesRegex(ValueError,
                                r"^.*Inputs .* not found in encoding.*$"):
      basic_encoder.encode([_BOS_TOKEN, 1, 2, 3, 4])

  def test_decode_raises_value_error_if_id_outside_of_vocab_size(self):
    vs = bases.VectorSpaceWithBasis.from_values("input", {1, 2, _BOS_TOKEN})
    basic_encoder = encoder.CategoricalEncoder(
        vs.basis, enforce_bos=True, bos_token=_BOS_TOKEN)
    with self.assertRaisesRegex(ValueError,
                                r"^.*Inputs .* not found in decoding map.*$"):
      basic_encoder.decode([0, 1, 2, 3])

  def test_encoder_raises_value_error_if_bos_not_in_basis(self):
    vs = bases.VectorSpaceWithBasis.from_values("input", {1, 2, 3})
    with self.assertRaisesRegex(ValueError,
                                r"^.*BOS token missing in encoding.*$"):
      unused_basic_encoder = encoder.CategoricalEncoder(
          vs.basis, bos_token=_BOS_TOKEN)

  def test_encoder_raises_value_error_if_pad_not_in_basis(self):
    vs = bases.VectorSpaceWithBasis.from_values("input", {1, 2, 3})
    with self.assertRaisesRegex(ValueError,
                                r"^.*PAD token missing in encoding.*$"):
      unused_basic_encoder = encoder.CategoricalEncoder(
          vs.basis, pad_token=_PAD_TOKEN)

  def test_encoder_encodes_bos_and_pad_tokens_as_expected(self):
    vs = bases.VectorSpaceWithBasis.from_values(
        "input", {1, 2, 3, _BOS_TOKEN, _PAD_TOKEN})
    basic_encoder = encoder.CategoricalEncoder(
        vs.basis, bos_token=_BOS_TOKEN, pad_token=_PAD_TOKEN)
    self.assertEqual(
        basic_encoder.encode([_BOS_TOKEN, _PAD_TOKEN]),
        [basic_encoder.bos_encoding, basic_encoder.pad_encoding])

  @parameterized.parameters([
      dict(
          vocab={1, 2, 3, _BOS_TOKEN},  # lexicographic order
          inputs=[_BOS_TOKEN, 3, 2, 1],
          expected=[3, 2, 1, 0]),
      dict(
          vocab={"a", "b", _BOS_TOKEN, "c"},  # lexicographic order
          inputs=[_BOS_TOKEN, "b", "b", "c"],
          expected=[2, 1, 1, 3]),
  ])
  def test_tokens_are_encoded_in_lexicographic_order(self, vocab, inputs,
                                                     expected):
    # Expect encodings to be assigned to ids according to a lexicographic
    # ordering of the vocab
    vs = bases.VectorSpaceWithBasis.from_values("input", vocab)
    basic_encoder = encoder.CategoricalEncoder(
        vs.basis, enforce_bos=True, bos_token=_BOS_TOKEN)
    encodings = basic_encoder.encode(inputs)
    self.assertEqual(encodings, expected)

  @parameterized.parameters([
      dict(vocab={_BOS_TOKEN, _PAD_TOKEN, 1, 2, 3}, expected=5),
      dict(vocab={_BOS_TOKEN, _PAD_TOKEN, "a", "b"}, expected=4),
  ])
  def test_vocab_size_has_expected_value(self, vocab, expected):
    vs = bases.VectorSpaceWithBasis.from_values("input", vocab)
    basic_encoder = encoder.CategoricalEncoder(
        vs.basis, enforce_bos=True, bos_token=_BOS_TOKEN, pad_token=_PAD_TOKEN)
    self.assertEqual(basic_encoder.vocab_size, expected)

  @parameterized.parameters([
      dict(
          vocab={_BOS_TOKEN, _PAD_TOKEN, 1, 2, 3}, inputs=[_BOS_TOKEN, 3, 2,
                                                           1]),
      dict(
          vocab={_BOS_TOKEN, _PAD_TOKEN, "a", "b", "c"},
          inputs=[_BOS_TOKEN, "b", "b", "c"]),
  ])
  def test_decode_inverts_encode(self, vocab, inputs):
    vs = bases.VectorSpaceWithBasis.from_values("input", vocab)
    basic_encoder = encoder.CategoricalEncoder(
        vs.basis, enforce_bos=True, bos_token=_BOS_TOKEN, pad_token=_PAD_TOKEN)
    encodings = basic_encoder.encode(inputs)
    recovered = basic_encoder.decode(encodings)
    self.assertEqual(recovered, inputs)


if __name__ == "__main__":
  absltest.main()
