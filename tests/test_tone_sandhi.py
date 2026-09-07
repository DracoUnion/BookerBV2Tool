"""Unit tests for ``text/tone_sandhi.py`` — the Mandarin tone-sandhi rules."""

import pytest

from BookerBV2Tool.text.tone_sandhi import ToneSandhi


@pytest.fixture(scope="module")
def ts():
    return ToneSandhi()


class TestBuSandhi:
    def test_bu_before_tone4(self, ts):
        assert ts._bu_sandhi("不怕", ["bu4", "pa4"]) == ["bu2", "pa4"]

    def test_bu_in_middle_word(self, ts):
        assert ts._bu_sandhi("看不懂", ["kan4", "bu4", "dong3"]) == [
            "kan4", "bu5", "dong3",
        ]


class TestYiSandhi:
    def test_number_sequence_unchanged(self, ts):
        assert ts._yi_sandhi("一零零", ["yi1", "ling2", "ling2"]) == [
            "yi1", "ling2", "ling2",
        ]

    def test_reduplication_coverb(self, ts):
        assert ts._yi_sandhi("看一看", ["kan4", "yi1", "kan4"]) == [
            "kan4", "yi5", "kan4",
        ]

    def test_ordinal_first(self, ts):
        assert ts._yi_sandhi("第一天", ["di4", "yi1", "tian1"]) == [
            "di4", "yi1", "tian1",
        ]

    def test_before_non_tone4(self, ts):
        assert ts._yi_sandhi("一天", ["yi1", "tian1"]) == ["yi4", "tian1"]

    def test_before_tone4(self, ts):
        assert ts._yi_sandhi("一段", ["yi1", "duan4"]) == ["yi2", "duan4"]


class TestNeuralSandhi:
    def test_reduplication_noun_verb(self, ts):
        assert ts._neural_sandhi("看看", "v", ["kan4", "kan4"]) == ["kan4", "kan5"]

    def test_word_in_must_neural_list(self, ts):
        assert ts._neural_sandhi("朋友", "n", ["peng2", "you3"]) == ["peng2", "you5"]

    def test_word_not_in_must_not_list_for_zi(self, ts):
        # 们/子 after a noun -> neutral
        assert ts._neural_sandhi("孩子们", "n", ["hai2", "zi5", "men5"])[-1] == "men5"


class TestThreeSandhi:
    def test_two_char_both_tone_three(self, ts):
        assert ts._three_sandhi("很好", ["hao3", "hao3"]) == ["hao2", "hao3"]

    def test_four_char_all_tone_three(self, ts):
        assert ts._three_sandhi("abcd", ["a3", "b3", "c3", "d3"]) == [
            "a2", "b3", "c2", "d3",
        ]

    def test_three_char_split(self, ts):
        # all-tone-three disyllabic+monosyllabic: first two become tone 2
        out = ts._three_sandhi("蒙古包", ["meng3", "gu3", "bao1"])
        # 蒙古 both tone 3 -> first two lowered to 2
        assert out[0].startswith("meng2")
        assert out[1].startswith("gu3")


class TestAllToneThree:
    def test_true(self, ts):
        assert ts._all_tone_three(["a3", "b3"]) is True

    def test_false(self, ts):
        assert ts._all_tone_three(["a3", "b1"]) is False


class TestMergeHelpers:
    def test_merge_bu(self, ts):
        assert ts._merge_bu([("不", "d"), ("怕", "v"), ("他", "r")]) == [
            ("不怕", "v"),
            ("他", "r"),
        ]

    def test_merge_yi_reduplication(self, ts):
        assert ts._merge_yi([("听", "v"), ("一", "m"), ("听", "v")]) == [["听一听", "v"]]

    def test_merge_yi_simple(self, ts):
        assert ts._merge_yi([("一", "m"), ("天", "q")]) == [["一天", "m"]]

    def test_merge_er(self, ts):
        assert ts._merge_er([("花", "n"), ("儿", "r")]) == [["花儿", "n"]]

    def test_merge_reduplication(self, ts):
        assert ts._merge_reduplication([("走", "v"), ("走", "v")]) == [["走走", "v"]]

    def test_is_reduplication(self, ts):
        assert ts._is_reduplication("走走") is True
        assert ts._is_reduplication("朋友") is False


class TestPreMerge:
    def test_end_to_end_reduplication_verb(self, ts):
        seg = ts.pre_merge_for_modify([("听", "v"), ("一", "m"), ("听", "v")])
        assert ["".join(w) for w, _ in seg] == ["听一听"]

    def test_merge_continuous_three_tones(self, ts):
        seg = ts._merge_continuous_three_tones([("好", "a"), ("我", "r")])
        assert ["".join(w) for w, _ in seg] == ["好我"]


class TestModifiedTone:
    def test_bu_and_yi_applied(self, ts):
        out = ts.modified_tone("不怕", "d", ["bu4", "pa4"])
        assert out == ["bu2", "pa4"]

    def test_does_not_crash_on_single_char(self, ts):
        out = ts.modified_tone("好", "a", ["hao3"])
        assert len(out) == 1