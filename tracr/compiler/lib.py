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
"""RASP programs only using the subset of RASP supported by the compiler."""

from typing import List, Sequence

from tracr.rasp import rasp

### Programs that work only under non-causal evaluation.


def make_length() -> rasp.SOp:
  """Creates the `length` SOp using selector width primitive.

  Example usage:
    length = make_length()
    length("abcdefg")
    >> [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]

  Returns:
    length: SOp mapping an input to a sequence, where every element
      is the length of that sequence.
  """
  all_true_selector = rasp.Select(
      rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")
  return rasp.SelectorWidth(all_true_selector).named("length")


length = make_length()


def make_reverse(sop: rasp.SOp) -> rasp.SOp:
  """Create an SOp that reverses a sequence, using length primitive.

  Example usage:
    reverse = make_reverse(rasp.tokens)
    reverse("Hello")
    >> ['o', 'l', 'l', 'e', 'H']

  Args:
    sop: an SOp

  Returns:
    reverse : SOp that reverses the input sequence.
  """
  opp_idx = (length - rasp.indices).named("opp_idx")
  opp_idx = (opp_idx - 1).named("opp_idx-1")
  reverse_selector = rasp.Select(rasp.indices, opp_idx,
                                 rasp.Comparison.EQ).named("reverse_selector")
  return rasp.Aggregate(reverse_selector, sop).named("reverse")


def make_pair_balance(sop: rasp.SOp, open_token: str,
                      close_token: str) -> rasp.SOp:
  """Return fraction of previous open tokens minus the fraction of close tokens.

   (As implemented in the RASP paper.)

  If the outputs are always non-negative and end in 0, that implies the input
  has balanced parentheses.

  Example usage:
    num_l = make_pair_balance(rasp.tokens, "(", ")")
    num_l("a()b(c))")
    >> [0, 1/2, 0, 0, 1/5, 1/6, 0, -1/8]

  Args:
    sop: Input SOp.
    open_token: Token that counts positive.
    close_token: Token that counts negative.

  Returns:
    pair_balance: SOp mapping an input to a sequence, where every element
      is the fraction of previous open tokens minus previous close tokens.
  """
  bools_open = rasp.numerical(sop == open_token).named("bools_open")
  opens = rasp.numerical(make_frac_prevs(bools_open)).named("opens")

  bools_close = rasp.numerical(sop == close_token).named("bools_close")
  closes = rasp.numerical(make_frac_prevs(bools_close)).named("closes")

  pair_balance = rasp.numerical(rasp.LinearSequenceMap(opens, closes, 1, -1))
  return pair_balance.named("pair_balance")


def make_shuffle_dyck(pairs: List[str]) -> rasp.SOp:
  """Returns 1 if a set of parentheses are balanced, 0 else.

   (As implemented in the RASP paper.)

  Example usage:
    shuffle_dyck2 = make_shuffle_dyck(pairs=["()", "{}"])
    shuffle_dyck2("({)}")
    >> [1, 1, 1, 1]
    shuffle_dyck2("(){)}")
    >> [0, 0, 0, 0, 0]

  Args:
    pairs: List of pairs of open and close tokens that each should be balanced.
  """
  assert len(pairs) >= 1

  # Compute running balance of each type of parenthesis
  balances = []
  for pair in pairs:
    assert len(pair) == 2
    open_token, close_token = pair
    balance = make_pair_balance(
        rasp.tokens, open_token=open_token,
        close_token=close_token).named(f"balance_{pair}")
    balances.append(balance)

  # Check if balances where negative anywhere -> parentheses not balanced
  any_negative = balances[0] < 0
  for balance in balances[1:]:
    any_negative = any_negative | (balance < 0)

  # Convert to numerical SOp
  any_negative = rasp.numerical(rasp.Map(lambda x: x,
                                         any_negative)).named("any_negative")

  select_all = rasp.Select(rasp.indices, rasp.indices,
                           rasp.Comparison.TRUE).named("select_all")
  has_neg = rasp.numerical(rasp.Aggregate(select_all, any_negative,
                                          default=0)).named("has_neg")

  # Check if all balances are 0 at the end -> closed all parentheses
  all_zero = balances[0] == 0
  for balance in balances[1:]:
    all_zero = all_zero & (balance == 0)

  select_last = rasp.Select(rasp.indices, length - 1,
                            rasp.Comparison.EQ).named("select_last")
  last_zero = rasp.Aggregate(select_last, all_zero).named("last_zero")

  not_has_neg = (~has_neg).named("not_has_neg")
  return (last_zero & not_has_neg).named("shuffle_dyck")


def make_shuffle_dyck2() -> rasp.SOp:
  return make_shuffle_dyck(pairs=["()", "{}"]).named("shuffle_dyck2")


def make_hist() -> rasp.SOp:
  """Returns the number of times each token occurs in the input.

   (As implemented in the RASP paper.)

  Example usage:
    hist = make_hist()
    hist("abac")
    >> [2, 1, 2, 1]
  """
  same_tok = rasp.Select(rasp.tokens, rasp.tokens,
                         rasp.Comparison.EQ).named("same_tok")
  return rasp.SelectorWidth(same_tok).named("hist")


def make_sort_unique(vals: rasp.SOp, keys: rasp.SOp) -> rasp.SOp:
  """Returns vals sorted by < relation on keys.

  Only supports unique keys.

  Example usage:
    sort = make_sort(rasp.tokens, rasp.tokens)
    sort([2, 4, 3, 1])
    >> [1, 2, 3, 4]

  Args:
    vals: Values to sort.
    keys: Keys for sorting.
  """
  smaller = rasp.Select(keys, keys, rasp.Comparison.LT).named("smaller")
  target_pos = rasp.SelectorWidth(smaller).named("target_pos")
  sel_new = rasp.Select(target_pos, rasp.indices, rasp.Comparison.EQ)
  return rasp.Aggregate(sel_new, vals).named("sort")


def make_sort(vals: rasp.SOp, keys: rasp.SOp, *, max_seq_len: int,
              min_key: float) -> rasp.SOp:
  """Returns vals sorted by < relation on keys, which don't need to be unique.

  The implementation differs from the RASP paper, as it avoids using
  compositions of selectors to break ties. Instead, it uses the arguments
  max_seq_len and min_key to ensure the keys are unique.

  Note that this approach only works for numerical keys.

  Example usage:
    sort = make_sort(rasp.tokens, rasp.tokens, 5, 1)
    sort([2, 4, 3, 1])
    >> [1, 2, 3, 4]
    sort([2, 4, 1, 2])
    >> [1, 2, 2, 4]

  Args:
    vals: Values to sort.
    keys: Keys for sorting.
    max_seq_len: Maximum sequence length (used to ensure keys are unique)
    min_key: Minimum key value (used to ensure keys are unique)

  Returns:
    Output SOp of sort program.
  """
  keys = rasp.SequenceMap(lambda x, i: x + min_key * i / max_seq_len, keys,
                          rasp.indices)
  return make_sort_unique(vals, keys)


def make_sort_freq(max_seq_len: int) -> rasp.SOp:
  """Returns tokens sorted by the frequency they appear in the input.

  Tokens the appear the same amount of times are output in the same order as in
  the input.

  Example usage:
    sort = make_sort_freq(rasp.tokens, rasp.tokens, 5)
    sort([2, 4, 2, 1])
    >> [2, 2, 4, 1]

  Args:
    max_seq_len: Maximum sequence length (used to ensure keys are unique)
  """
  hist = -1 * make_hist().named("hist")
  return make_sort(
      rasp.tokens, hist, max_seq_len=max_seq_len, min_key=1).named("sort_freq")


### Programs that work under both causal and regular evaluation.


def make_frac_prevs(bools: rasp.SOp) -> rasp.SOp:
  """Count the fraction of previous tokens where a specific condition was True.

   (As implemented in the RASP paper.)

  Example usage:
    num_l = make_frac_prevs(rasp.tokens=="l")
    num_l("hello")
    >> [0, 0, 1/3, 1/2, 2/5]

  Args:
    bools: SOp mapping a sequence to a sequence of booleans.

  Returns:
    frac_prevs: SOp mapping an input to a sequence, where every element
      is the fraction of previous "True" tokens.
  """
  bools = rasp.numerical(bools)
  prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
  return rasp.numerical(rasp.Aggregate(prevs, bools,
                                       default=0)).named("frac_prevs")


def shift_by(offset: int, /, sop: rasp.SOp) -> rasp.SOp:
  """Returns the sop, shifted by `offset`, None-padded."""
  select_off_by_offset = rasp.Select(rasp.indices, rasp.indices,
                                     lambda k, q: q == k + offset)
  out = rasp.Aggregate(select_off_by_offset, sop, default=None)
  return out.named(f"shift_by({offset})")


def detect_pattern(sop: rasp.SOp, pattern: Sequence[rasp.Value]) -> rasp.SOp:
  """Returns an SOp which is True at the final element of the pattern.

  The first len(pattern) - 1 elements of the output SOp are None-padded.

  detect_pattern(tokens, "abc")("abcabc") == [None, None, T, F, F, T]

  Args:
    sop: the SOp in which to look for patterns.
    pattern: a sequence of values to look for.

  Returns:
    a sop which detects the pattern.
  """

  if len(pattern) < 1:
    raise ValueError(f"Length of `pattern` must be at least 1. Got {pattern}")

  # detectors[i] will be a boolean-valued SOp which is true at position j iff
  # the i'th (from the end) element of the pattern was detected at position j-i.
  detectors = []
  for i, element in enumerate(reversed(pattern)):
    detector = sop == element
    if i != 0:
      detector = shift_by(i, detector)
    detectors.append(detector)

  # All that's left is to take the AND over all detectors.
  pattern_detected = detectors.pop()
  while detectors:
    pattern_detected = pattern_detected & detectors.pop()

  return pattern_detected.named(f"detect_pattern({pattern})")


def make_count_less_freq(n: int) -> rasp.SOp:
  """Returns how many tokens appear fewer than n times in the input.

  The output sequence contains this count in each position.

  Example usage:
    count_less_freq = make_count_less_freq(2)
    count_less_freq(["a", "a", "a", "b", "b", "c"])
    >> [3, 3, 3, 3, 3, 3]
    count_less_freq(["a", "a", "c", "b", "b", "c"])
    >> [6, 6, 6, 6, 6, 6]

  Args:
    n: Integer to compare token frequences to.
  """
  hist = make_hist().named("hist")
  select_less = rasp.Select(hist, hist,
                            lambda x, y: x <= n).named("select_less")
  return rasp.SelectorWidth(select_less).named("count_less_freq")


def make_count(sop, token):
  """Returns the count of `token` in `sop`.

  The output sequence contains this count in each position.

  Example usage:
    count = make_count(tokens, "a")
    count(["a", "a", "a", "b", "b", "c"])
    >> [3, 3, 3, 3, 3, 3]
    count(["c", "a", "b", "c"])
    >> [1, 1, 1, 1]

  Args:
    sop: Sop to count tokens in.
    token: Token to count.
  """
  return rasp.SelectorWidth(rasp.Select(
      sop, sop, lambda k, q: k == token)).named(f"count_{token}")


def make_nary_sequencemap(f, *sops):
  """Returns an SOp that simulates an n-ary SequenceMap.

  Uses multiple binary SequenceMaps to convert n SOps x_1, x_2, ..., x_n
  into a single SOp arguments that takes n-tuples as value. The n-ary sequence
  map implementing f is then a Map on this resulting SOp.

  Note that the intermediate variables representing tuples of varying length
  will be encoded categorically, and can become very high-dimensional. So,
  using this function might lead to very large compiled models.

  Args:
    f: Function with n arguments.
    *sops: Sequence of SOps, one for each argument of f.
  """
  values, *sops = sops
  for sop in sops:
    # x is a single entry in the first iteration but a tuple in later iterations
    values = rasp.SequenceMap(
        lambda x, y: (*x, y) if isinstance(x, tuple) else (x, y), values, sop)
  return rasp.Map(lambda args: f(*args), values)
