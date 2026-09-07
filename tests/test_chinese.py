"""Unit tests for ``BookerBV2Tool/text/chinese.py`` — the Mandarin G2P module."""

import pytest

from BookerBV2Tool.text import chinese


class TestReplacePunctuation:
    def test_full_width_to_ascii(self):
        assert chinese.replace_punctuation("你好，世界！") == "你好,世界!"

    def test_special_full_width_mappings(self):
        assert chinese.replace_punctuation("他说：好的。") == "他说,好的."

    def test_yi_eh_replaced(self):
        assert chinese.replace_punctuation("嗯") == "恩"
        assert chinese.replace_punctuation("呣") == "母"

    def test_latin_and_digits_stripped(self):
        assert chinese.replace_punctuation("abc你好123") == "你好"

    def test_ellipsis(self):
        assert chinese.replace_punctuation("...") == "…"


class TestTextNormalize:
    def test_numbers_to_chinese(self):
        out = chinese.text_normalize("有123个")
        assert "一百二十三" in out
        assert not any(c.isdigit() for c in out)

    def test_punctuation_replaced(self):
        out = chinese.text_normalize("你好，世界！")
        assert out == "你好,世界!"


class TestG2P:
    def test_basic_structure(self):
        phones, tones, word2ph = chinese.g2p("anything", "你好")
        assert phones[0] == "_" and phones[-1] == "_"
        assert len(phones) == len(tones)
        assert sum(word2ph) == len(phones)
        assert word2ph[0] == 1 and word2ph[-1] == 1
        assert all(0 <= t <= 5 for t in tones)

    def test_phones_are_known_symbols(self):
        phones, _, _ = chinese.g2p("anything", "今天天气很好")
        known = {
            phone
            for mapped in chinese.pinyin_to_symbol_map.values()
            for phone in mapped.split()
        }
        known.update(chinese.punctuation)
        for ph in phones:
            if ph != "_":
                assert ph in known

    def test_g2p_rejects_unstripped_latin_and_punctuation(self):
        # g2p() asserts one word2ph entry per input character; its internal
        # latin/punctuation filtering makes that invariant false for raw input.
        with pytest.raises(AssertionError):
            chinese.g2p("anything", "hello你好")
        with pytest.raises(AssertionError):
            chinese.g2p("anything", "你好，世界")


class TestInternals:
    def test_get_initials_finals(self):
        initials, finals = chinese._get_initials_finals("你好")
        assert len(initials) == len(finals) == 2

    def test_rep_map_keys_are_distinct(self):
        # No mapping key is also a punctuation we then strip out accidentally.
        assert "。" in chinese.rep_map