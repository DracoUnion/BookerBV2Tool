"""Unit tests for the offline-safe helpers in ``text/japanese.py``.

The ``g2p`` / ``text2sep_kata`` entry points require ``pyopenjtalk``'s
dictionary, which must be downloaded on first use, so they are not exercised
here; the pure text transforms below are fully self-contained.
"""

import pytest

from BookerBV2Tool.text import japanese


class TestHiragana2p:
    def test_basic_word(self):
        assert japanese.hiragana2p("こんにちは") == "k o n n i ch i h a"

    def test_long_vowel_expansion(self):
        # あーー -> a a a
        assert japanese.hiragana2p("あーー") == "a a a"

    def test_dakuon_v(self):
        assert japanese.hiragana2p("ゔ") == "v u"

    def test_geminate_to_q_then_n(self):
        # っ maps to q; ん maps to N then lowercased to n
        assert "q" in japanese.hiragana2p("あった")


class TestKata2Phoneme:
    def test_katakana_word(self):
        assert japanese.kata2phoneme("コンニチハ") == [
            "k", "o", "n", "n", "i", "ch", "i", "h", "a",
        ]

    def test_katakana_long_vowel(self):
        assert japanese.kata2phoneme("あーー") == ["a", "a", "a"]

    def test_hiragana_input(self):
        assert japanese.kata2phoneme("こんにちは") == [
            "k", "o", "n", "n", "i", "ch", "i", "h", "a",
        ]

    def test_latin_returns_empty_entry(self):
        # Characterization: unrecognised latin input yields a single "" entry.
        assert japanese.kata2phoneme("ABC") == [""]


class TestNumberAndAlphaConversion:
    def test_numbers(self):
        assert japanese.japanese_convert_numbers_to_words("100") == "百"
        assert japanese.japanese_convert_numbers_to_words("1,000") == "千"
        assert japanese.japanese_convert_numbers_to_words("12.5") == "十二点五"

    def test_currency(self):
        assert japanese.japanese_convert_numbers_to_words("$5") == "五ドル"

    def test_alpha_symbols(self):
        assert japanese.japanese_convert_alpha_symbols_to_words("Hello") == (
            "エイチイーエルエルオー"
        )

    def test_alpha_unknown_chars_kept(self):
        assert japanese.japanese_convert_alpha_symbols_to_words("a1") == "エー1"


class TestTextNormalize:
    def test_combines_all_steps(self):
        assert japanese.text_normalize("1,000円、ABC！") == "千円,!"

    def test_numbers_in_sentence(self):
        assert japanese.text_normalize("100円") == "百円"

    def test_nfkc_combining_dakuten_removed(self):
        assert "゙" not in japanese.text_normalize("だ")


class TestCharacterClassification:
    def test_hiragana(self):
        assert japanese.is_japanese_character("あ") is True

    def test_kanji(self):
        assert japanese.is_japanese_character("漢") is True

    def test_latin(self):
        assert japanese.is_japanese_character("a") is False

    def test_katakana(self):
        assert japanese.is_japanese_character("ア") is True


class TestWordHelpers:
    def test_distribute_phone(self):
        assert japanese.distribute_phone(3, 2) == [2, 1]
        assert japanese.distribute_phone(4, 2) == [2, 2]

    def test_handle_long_repeats_previous(self):
        assert japanese.handle_long([["k", "o"], ["ー"]]) == [["k", "o"], ["o"]]

    def test_align_tones_preserves_first_accent(self):
        assert japanese.align_tones([["k", "o", "n", "i"]], [("k", 1)]) == [0, 1, 0, 0]

    def test_align_tones_bounds(self):
        # output must only ever be 0 or 1
        out = japanese.align_tones(
            [["k", "o"], ["n", "i"]], [("o", 1), ("i", 0)]
        )
        assert all(v in (0, 1) for v in out)

    def test_replace_punctuation_full_width(self):
        assert japanese.replace_punctuation("こんにちは、世界！") == "こんにちは,世界!"

    def test_replace_punctuation_removes_unknown_chars(self):
        out = japanese.replace_punctuation("ABCこんにちは")
        assert "A" not in out and "こんにちは" in out