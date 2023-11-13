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
"""A set of RASP programs and input/output pairs used in integration tests."""

from tracr.compiler import lib
from tracr.rasp import rasp

UNIVERSAL_TEST_CASES = [
    dict(
        testcase_name="frac_prevs_1",
        program=lib.make_frac_prevs(rasp.tokens == "l"),
        vocab={"h", "e", "l", "o"},
        test_input=list("hello"),
        expected_output=[0.0, 0.0, 1 / 3, 1 / 2, 2 / 5],
        max_seq_len=5,
    ),
    dict(
        testcase_name="frac_prevs_2",
        program=lib.make_frac_prevs(rasp.tokens == "("),
        vocab={"a", "b", "c", "(", ")"},
        test_input=list("a()b(c))"),
        expected_output=[0.0, 1 / 2, 1 / 3, 1 / 4, 2 / 5, 2 / 6, 2 / 7, 2 / 8],
        max_seq_len=10,
    ),
    dict(
        testcase_name="frac_prevs_3",
        program=lib.make_frac_prevs(rasp.tokens == ")"),
        vocab={"a", "b", "c", "(", ")"},
        test_input=list("a()b(c))"),
        expected_output=[0.0, 0.0, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 2 / 7, 3 / 8],
        max_seq_len=10,
    ),
    dict(
        testcase_name="shift_by_one",
        program=lib.shift_by(1, rasp.tokens),
        vocab={"a", "b", "c", "d"},
        test_input=list("abcd"),
        expected_output=[None, "a", "b", "c"],
        max_seq_len=5,
    ),
    dict(
        testcase_name="shift_by_two",
        program=lib.shift_by(2, rasp.tokens),
        vocab={"a", "b", "c", "d"},
        test_input=list("abcd"),
        expected_output=[None, None, "a", "b"],
        max_seq_len=5,
    ),
    dict(
        testcase_name="detect_pattern_a",
        program=lib.detect_pattern(rasp.tokens, "a"),
        vocab={"a", "b", "c", "d"},
        test_input=list("bacd"),
        expected_output=[False, True, False, False],
        max_seq_len=5,
    ),
    dict(
        testcase_name="detect_pattern_ab",
        program=lib.detect_pattern(rasp.tokens, "ab"),
        vocab={"a", "b"},
        test_input=list("aaba"),
        expected_output=[None, False, True, False],
        max_seq_len=5,
    ),
    dict(
        testcase_name="detect_pattern_ab_2",
        program=lib.detect_pattern(rasp.tokens, "ab"),
        vocab={"a", "b"},
        test_input=list("abaa"),
        expected_output=[None, True, False, False],
        max_seq_len=5,
    ),
    dict(
        testcase_name="detect_pattern_ab_3",
        program=lib.detect_pattern(rasp.tokens, "ab"),
        vocab={"a", "b"},
        test_input=list("aaaa"),
        expected_output=[None, False, False, False],
        max_seq_len=5,
    ),
    dict(
        testcase_name="detect_pattern_abc",
        program=lib.detect_pattern(rasp.tokens, "abc"),
        vocab={"a", "b", "c"},
        test_input=list("abcabc"),
        expected_output=[None, None, True, False, False, True],
        max_seq_len=6,
    ),
]

TEST_CASES = UNIVERSAL_TEST_CASES + [
    dict(
        testcase_name="reverse_1",
        program=lib.make_reverse(rasp.tokens),
        vocab={"a", "b", "c", "d"},
        test_input=list("abcd"),
        expected_output=list("dcba"),
        max_seq_len=5,
    ),
    dict(
        testcase_name="reverse_2",
        program=lib.make_reverse(rasp.tokens),
        vocab={"a", "b", "c", "d"},
        test_input=list("abc"),
        expected_output=list("cba"),
        max_seq_len=5,
    ),
    dict(
        testcase_name="reverse_3",
        program=lib.make_reverse(rasp.tokens),
        vocab={"a", "b", "c", "d"},
        test_input=list("ad"),
        expected_output=list("da"),
        max_seq_len=5,
    ),
    dict(
        testcase_name="reverse_4",
        program=lib.make_reverse(rasp.tokens),
        vocab={"a", "b", "c", "d"},
        test_input=["c"],
        expected_output=["c"],
        max_seq_len=5,
    ),
    dict(
        testcase_name="length_categorical_1",
        program=rasp.categorical(lib.make_length()),
        vocab={"a", "b", "c", "d"},
        test_input=list("abc"),
        expected_output=[3, 3, 3],
        max_seq_len=3,
    ),
    dict(
        testcase_name="length_categorical_2",
        program=rasp.categorical(lib.make_length()),
        vocab={"a", "b", "c", "d"},
        test_input=list("ad"),
        expected_output=[2, 2],
        max_seq_len=3,
    ),
    dict(
        testcase_name="length_categorical_3",
        program=rasp.categorical(lib.make_length()),
        vocab={"a", "b", "c", "d"},
        test_input=["c"],
        expected_output=[1],
        max_seq_len=3,
    ),
    dict(
        testcase_name="length_numerical_1",
        program=rasp.numerical(lib.make_length()),
        vocab={"a", "b", "c", "d"},
        test_input=list("abc"),
        expected_output=[3, 3, 3],
        max_seq_len=3,
    ),
    dict(
        testcase_name="length_numerical_2",
        program=rasp.numerical(lib.make_length()),
        vocab={"a", "b", "c", "d"},
        test_input=list("ad"),
        expected_output=[2, 2],
        max_seq_len=3,
    ),
    dict(
        testcase_name="length_numerical_3",
        program=rasp.numerical(lib.make_length()),
        vocab={"a", "b", "c", "d"},
        test_input=["c"],
        expected_output=[1],
        max_seq_len=3,
    ),
    dict(
        testcase_name="pair_balance_1",
        program=lib.make_pair_balance(rasp.tokens, "(", ")"),
        vocab={"a", "b", "c", "(", ")"},
        test_input=list("a()b(c))"),
        expected_output=[0.0, 1 / 2, 0.0, 0.0, 1 / 5, 1 / 6, 0.0, -1 / 8],
        max_seq_len=10,
    ),
    dict(
        testcase_name="shuffle_dyck2_1",
        program=lib.make_shuffle_dyck(pairs=["()", "{}"]),
        vocab={"(", ")", "{", "}"},
        test_input=list("({)}"),
        expected_output=[1, 1, 1, 1],
        max_seq_len=5,
    ),
    dict(
        testcase_name="shuffle_dyck2_2",
        program=lib.make_shuffle_dyck(pairs=["()", "{}"]),
        vocab={"(", ")", "{", "}"},
        test_input=list("(){)}"),
        expected_output=[0, 0, 0, 0, 0],
        max_seq_len=5,
    ),
    dict(
        testcase_name="shuffle_dyck2_3",
        program=lib.make_shuffle_dyck(pairs=["()", "{}"]),
        vocab={"(", ")", "{", "}"},
        test_input=list("{}("),
        expected_output=[0, 0, 0],
        max_seq_len=5,
    ),
    dict(
        testcase_name="shuffle_dyck3_1",
        program=lib.make_shuffle_dyck(pairs=["()", "{}", "[]"]),
        vocab={"(", ")", "{", "}", "[", "]"},
        test_input=list("({)[}]"),
        expected_output=[1, 1, 1, 1, 1, 1],
        max_seq_len=6,
    ),
    dict(
        testcase_name="shuffle_dyck3_2",
        program=lib.make_shuffle_dyck(pairs=["()", "{}", "[]"]),
        vocab={"(", ")", "{", "}", "[", "]"},
        test_input=list("(){)}"),
        expected_output=[0, 0, 0, 0, 0],
        max_seq_len=6,
    ),
    dict(
        testcase_name="shuffle_dyck3_3",
        program=lib.make_shuffle_dyck(pairs=["()", "{}", "[]"]),
        vocab={"(", ")", "{", "}", "[", "]"},
        test_input=list("{}[(]"),
        expected_output=[0, 0, 0, 0, 0],
        max_seq_len=6,
    ),
    dict(
        testcase_name="hist",
        program=lib.make_hist(),
        vocab={"a", "b", "c", "d"},
        test_input=list("abac"),
        expected_output=[2, 1, 2, 1],
        max_seq_len=5,
    ),
    dict(
        testcase_name="sort_unique_1",
        program=lib.make_sort_unique(vals=rasp.tokens, keys=rasp.tokens),
        vocab={1, 2, 3, 4},
        test_input=[2, 4, 3, 1],
        expected_output=[1, 2, 3, 4],
        max_seq_len=5,
    ),
    dict(
        testcase_name="sort_unique_2",
        program=lib.make_sort_unique(vals=rasp.tokens, keys=1 - rasp.indices),
        vocab={"a", "b", "c", "d"},
        test_input=list("abcd"),
        expected_output=["d", "c", "b", "a"],
        max_seq_len=5,
    ),
    dict(
        testcase_name="sort_1",
        program=lib.make_sort(
            vals=rasp.tokens, keys=rasp.tokens, max_seq_len=5, min_key=1
        ),
        vocab={1, 2, 3, 4},
        test_input=[2, 4, 3, 1],
        expected_output=[1, 2, 3, 4],
        max_seq_len=5,
    ),
    dict(
        testcase_name="sort_2",
        program=lib.make_sort(
            vals=rasp.tokens, keys=1 - rasp.indices, max_seq_len=5, min_key=1
        ),
        vocab={"a", "b", "c", "d"},
        test_input=list("abcd"),
        expected_output=["d", "c", "b", "a"],
        max_seq_len=5,
    ),
    dict(
        testcase_name="sort_3",
        program=lib.make_sort(
            vals=rasp.tokens, keys=rasp.tokens, max_seq_len=5, min_key=1
        ),
        vocab={1, 2, 3, 4},
        test_input=[2, 4, 1, 2],
        expected_output=[1, 2, 2, 4],
        max_seq_len=5,
    ),
    dict(
        testcase_name="sort_freq",
        program=lib.make_sort_freq(max_seq_len=5),
        vocab={1, 2, 3, 4},
        test_input=[2, 4, 2, 1],
        expected_output=[2, 2, 4, 1],
        max_seq_len=5,
    ),
    dict(
        testcase_name="make_count_less_freq_categorical_1",
        program=lib.make_count_less_freq(n=2),
        vocab={"a", "b", "c", "d"},
        test_input=["a", "a", "a", "b", "b", "c"],
        expected_output=[3, 3, 3, 3, 3, 3],
        max_seq_len=6,
    ),
    dict(
        testcase_name="make_count_less_freq_categorical_2",
        program=lib.make_count_less_freq(n=2),
        vocab={"a", "b", "c", "d"},
        test_input=["a", "a", "c", "b", "b", "c"],
        expected_output=[6, 6, 6, 6, 6, 6],
        max_seq_len=6,
    ),
    dict(
        testcase_name="make_count_less_freq_numerical_1",
        program=rasp.numerical(lib.make_count_less_freq(n=2)),
        vocab={"a", "b", "c", "d"},
        test_input=["a", "a", "a", "b", "b", "c"],
        expected_output=[3, 3, 3, 3, 3, 3],
        max_seq_len=6,
    ),
    dict(
        testcase_name="make_count_less_freq_numerical_2",
        program=rasp.numerical(lib.make_count_less_freq(n=2)),
        vocab={"a", "b", "c", "d"},
        test_input=["a", "a", "c", "b", "b", "c"],
        expected_output=[6, 6, 6, 6, 6, 6],
        max_seq_len=6,
    ),
    dict(
        testcase_name="make_count_1",
        program=lib.make_count(rasp.tokens, "a"),
        vocab={"a", "b", "c"},
        test_input=["a", "a", "a", "b", "b", "c"],
        expected_output=[3, 3, 3, 3, 3, 3],
        max_seq_len=8,
    ),
    dict(
        testcase_name="make_count_2",
        program=lib.make_count(rasp.tokens, "a"),
        vocab={"a", "b", "c"},
        test_input=["c", "a", "b", "c"],
        expected_output=[1, 1, 1, 1],
        max_seq_len=8,
    ),
    dict(
        testcase_name="make_count_3",
        program=lib.make_count(rasp.tokens, "a"),
        vocab={"a", "b", "c"},
        test_input=["b", "b", "c"],
        expected_output=[0, 0, 0],
        max_seq_len=8,
    ),
    dict(
        testcase_name="make_nary_sequencemap_1",
        program=lib.make_nary_sequencemap(
            lambda x, y, z: x + y - z, rasp.tokens, rasp.tokens, rasp.indices
        ),
        vocab={1, 2, 3},
        test_input=[1, 2, 3],
        expected_output=[2, 3, 4],
        max_seq_len=5,
    ),
    dict(
        testcase_name="make_nary_sequencemap_2",
        program=lib.make_nary_sequencemap(
            lambda x, y, z: x * y / z, rasp.indices, rasp.indices, rasp.tokens
        ),
        vocab={1, 2, 3},
        test_input=[1, 2, 3],
        expected_output=[0, 1 / 2, 4 / 3],
        max_seq_len=3,
    ),
]

# make_nary_sequencemap(f, *sops)

CAUSAL_TEST_CASES = UNIVERSAL_TEST_CASES + [
    dict(
        testcase_name="selector_width",
        program=rasp.SelectorWidth(
            rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
        ),
        vocab={"a", "b", "c", "d"},
        test_input=list("abcd"),
        expected_output=[1, 2, 3, 4],
        max_seq_len=5,
    ),
]


# Programs using features that are currently not supported by Tracr and that
# cause the compiler to throw NotImplementerError.
UNSUPPORTED_TEST_CASES = [
    dict(
        testcase_name="numerical_categorical_aggregate",
        program=rasp.Aggregate(
            rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE),
            rasp.numerical(rasp.Map(lambda x: x, rasp.tokens)),
        ),
        vocab={1, 2, 3},
        max_seq_len=5,
    ),
    dict(
        testcase_name="categorical_numerical_aggregate",
        program=rasp.numerical(
            rasp.Aggregate(
                rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE),
                rasp.tokens,
            ),
        ),
        vocab={1, 2, 3},
        max_seq_len=5,
    ),
    dict(
        testcase_name="numerical_numerical_aggregate",
        program=rasp.numerical(
            rasp.Aggregate(
                rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE),
                rasp.numerical(rasp.Map(lambda x: x, rasp.tokens)),
                default=0
            ),
        ),
        vocab={1, 2, 3},
        max_seq_len=5,
    ),
    dict(
        testcase_name="aggregate_with_not_None_default",
        program=rasp.numerical(
            rasp.Aggregate(
                rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.EQ),
                rasp.tokens,
                default=1
            ),
        ),
        vocab={1, 2, 3},
        max_seq_len=5,
    ),
    dict(
        testcase_name="numerical_selector",
        program=rasp.SelectorWidth(
            rasp.Select(
                rasp.numerical(rasp.Map(lambda x: x, rasp.tokens)),
                rasp.numerical(rasp.Map(lambda x: x, rasp.tokens)),
                rasp.Comparison.LT,
            )
        ),
        vocab={1, 2, 3},
        max_seq_len=5,
    ),
    dict(
        testcase_name="numerical_SequenceMap",
        program=rasp.numerical(
            rasp.SequenceMap(
                lambda x, y: x + y,
                rasp.numerical(rasp.Map(lambda x: x, rasp.indices)),
                rasp.numerical(rasp.Map(lambda x: x, rasp.tokens)),
            )
        ),
        vocab={1, 2, 3},
        max_seq_len=5,
    ),
    dict(
        testcase_name="categorical_LinearSequenceMap",
        program=rasp.categorical(
            rasp.LinearSequenceMap(
                rasp.categorical(rasp.indices),
                rasp.categorical(rasp.tokens),
                1,
                1,
            )
        ),
        vocab={1, 2, 3},
        max_seq_len=5,
    ),
    dict(
        testcase_name="numerical_tokens",
        program=rasp.numerical(rasp.tokens),
        vocab={1, 2, 3},
        max_seq_len=5,
    ),
    dict(
        testcase_name="numerical_indices",
        program=rasp.numerical(rasp.indices),
        vocab={1, 2, 3},
        max_seq_len=5,
    ),
]
