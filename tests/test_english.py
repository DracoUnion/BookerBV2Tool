"""Unit tests for the offline-safe helpers in ``text/english.py``."""

import re

import pytest

from BookerBV2Tool.text import english


class TestPhoneHelpers:
    def test_refine_ph_with_tone(self):
        assert english.refine_ph("AH0") == ("ah", 1)
        assert english.refine_ph("AH1") == ("ah", 2)
        assert english.refine_ph("OY2") == ("oy", 3)

    def test_refine_ph_without_tone(self):
        assert english.refine_ph("HH") == ("hh", 3)

    def test_refine_syllables(self):
        phones, tones = english.refine_syllables([["AH0", "S"], ["IY1"]])
        assert phones == ["ah", "s", "iy"]
        assert tones == [1, 3, 2]

    def test_post_replace_ph(self):
        assert english.post_replace_ph("v") == "V"
        assert english.post_replace_ph("V") == "V"
        assert english.post_replace_ph("not-a-phone") == "UNK"
        # punctuation is not part of english.en_symbols, so it becomes UNK
        assert english.post_replace_ph("，") == "UNK"


class TestNumberNormalization:
    def test_dollars(self):
        assert english.normalize_numbers("$1.25") == "one dollar, twenty-five cents"
        assert english.normalize_numbers("$2") == "two dollars"

    def test_comma_separated_numbers(self):
        # commas are removed first, then the number is spoken out
        assert english.normalize_numbers("1,000") == "one thousand"

    def test_decimals(self):
        assert english.normalize_numbers("3.5") == "three point five"

    def test_ordinal(self):
        assert english.normalize_numbers("12th") == "twelfth"

    def test_years_in_1000_to_3000(self):
        assert english.normalize_numbers("2020") == "twenty twenty"
        assert english.normalize_numbers("2000") == "two thousand"

    def test_pounds(self):
        # £ marker is expanded to "pounds", then the digit is spoken out
        assert english.normalize_numbers("£3") == "three pounds"

    def test_number_outside_year_range(self):
        assert english.normalize_numbers("42") == "forty-two"


class TestTextNormalization:
    def test_full_width_punctuation(self):
        assert english.replace_punctuation("你好，世界！") == "你好,世界!"

    def test_adds_space_after_punctuation(self):
        assert english.text_normalize("Hello,2020!") == "Hello, twenty twenty!"

    def test_existing_spaces_preserved(self):
        out = english.text_normalize("Hello world.")
        assert out == "Hello world."


class TestWordHelpers:
    def test_distribute_phone_balances_left_to_right(self):
        assert english.distribute_phone(3, 2) == [2, 1]
        assert english.distribute_phone(4, 2) == [2, 2]
        assert english.distribute_phone(0, 0) == []

    def test_sep_text_discards_whitespace_tokens(self):
        assert english.sep_text("abc def, ghi.") == ["abc", "def", ",", "ghi", "."]

    def test_expand_dollars_edge_cases(self):
        class M:
            def __init__(self, value):
                self.value = value

            def group(self, index):
                assert index == 1
                return self.value

        assert english._expand_dollars(M("1")) == "1 dollar"
        assert english._expand_dollars(M(".50")) == "50 cents"
        assert english._expand_dollars(M("")) == "zero dollars"
        assert english._expand_dollars(M("1.2.3")) == "1.2.3 dollars"

    def test_g2p_with_mocked_tokenizer_path(self, monkeypatch):
        # Avoid downloading a Transformer tokenizer while exercising the rest
        # of the G2P pipeline and its invariant checks.
        monkeypatch.setattr(
            english,
            "text_to_words",
            lambda model_name, text: [["hello"], ["hello"], ["."]],
        )
        phones, tones, word2ph = english.g2p("unused-model", "hello hello.")
        assert phones[0] == "_" and phones[-1] == "_"
        assert len(phones) == len(tones)
        assert sum(word2ph) == len(phones)
        assert "." in phones
        assert all(isinstance(t, int) for t in tones)