"""Unit tests for ``BookerBV2Tool/text/symbols.py`` — the shared phoneme tables."""

import pytest

from BookerBV2Tool.text import symbols


class TestSymbolTable:
    def test_pad_is_first(self):
        assert symbols.symbols[0] == "_"
        assert symbols.pad == "_"

    def test_no_duplicate_symbols(self):
        assert len(symbols.symbols) == len(set(symbols.symbols))

    def test_all_are_strings(self):
        assert all(isinstance(s, str) for s in symbols.symbols)

    def test_contains_phoneme_sets(self):
        for sym in ("a", "n", "ih", "sh", "a:", "N", "q"):
            assert sym in symbols.symbols

    def test_punctuation_is_in_symbols(self):
        for p in symbols.punctuation + ["SP", "UNK"]:
            assert p in symbols.symbols

    def test_sil_phoneme_ids(self):
        expected = [symbols.symbols.index(i) for i in symbols.pu_symbols]
        assert symbols.sil_phonemes_ids == expected

    def test_language_id_map(self):
        assert symbols.language_id_map == {"ZH": 0, "JP": 1, "EN": 2}
        assert symbols.num_languages == 3

    def test_tone_start_map(self):
        assert symbols.language_tone_start_map == {"ZH": 0, "JP": 6, "EN": 8}

    def test_num_tones_total(self):
        assert symbols.num_tones == symbols.num_zh_tones + symbols.num_ja_tones + symbols.num_en_tones

    def test_phoneme_counts(self):
        assert symbols.num_zh_tones == 6
        assert symbols.num_ja_tones == 2
        assert symbols.num_en_tones == 4