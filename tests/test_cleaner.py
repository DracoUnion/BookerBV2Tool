"""Unit tests for ``BookerBV2Tool/text/cleaner.py``."""

from types import SimpleNamespace

import pytest

from BookerBV2Tool.text import cleaner
from BookerBV2Tool.text import symbols


class TestCleanedTextToSequence:
    def test_chinese(self):
        phones, tones, langs = cleaner.cleaned_text_to_sequence(
            ["n", "ih", "sh"], [1, 2, 3], "ZH"
        )
        assert phones == [cleaner._symbol_to_id[s] for s in ["n", "ih", "sh"]]
        assert tones == [1, 2, 3]  # ZH tone offset is 0
        assert langs == [0, 0, 0]

    def test_japanese_tone_offset(self):
        phones, tones, langs = cleaner.cleaned_text_to_sequence(
            ["k", "o"], [1, 0], "JP"
        )
        assert tones == [1 + symbols.num_zh_tones, 0 + symbols.num_zh_tones]
        assert langs == [1, 1]

    def test_english_tone_offset(self):
        phones, tones, langs = cleaner.cleaned_text_to_sequence(
            ["ah", "hh"], [1, 3], "EN"
        )
        assert tones == [9, 11]  # 8 + tone
        assert langs == [2, 2]

    def test_unknown_symbol_raises_keyerror(self):
        with pytest.raises(KeyError):
            cleaner.cleaned_text_to_sequence(["not-a-phone"], [0], "ZH")


class TestGetModelNameByLang:
    def test_map(self):
        args = SimpleNamespace(chinese_bert="cb", enlish_bert="eb", japanese_bert="jb")
        assert cleaner.get_model_name_by_lang("ZH", args) == "cb"
        assert cleaner.get_model_name_by_lang("EN", args) == "eb"
        assert cleaner.get_model_name_by_lang("JP", args) == "jb"

    def test_unknown_lang(self):
        args = SimpleNamespace(chinese_bert="cb", enlish_bert="eb", japanese_bert="jb")
        assert cleaner.get_model_name_by_lang("XX", args) == ""


class TestCleanTextFlow:
    def test_clean_text_dispatches_to_language_module(self, monkeypatch):
        fake = lambda model_name, norm_text: (["a", "b"], [1, 2], [1, 1])
        monkeypatch.setattr(cleaner.chinese, "text_normalize", lambda t: "NORM")
        monkeypatch.setattr(cleaner.chinese, "g2p", fake)
        norm, phones, tones, word2ph = cleaner.clean_text("m", "原文", "ZH")
        assert norm == "NORM"
        assert phones == ["a", "b"]
        assert tones == [1, 2]
        assert word2ph == [1, 1]

    def test_clean_text_bert_calls_feature(self, monkeypatch):
        import torch

        calls = {}

        def fake_g2p(model_name, norm_text):
            return ["a", "b"], [1, 2], [1, 1]

        monkeypatch.setattr(cleaner.chinese, "text_normalize", lambda t: "NORM")
        monkeypatch.setattr(cleaner.chinese, "g2p", fake_g2p)
        monkeypatch.setattr(
            cleaner, "get_bert_feature", lambda *a, **k: torch.zeros(2, 2)
        )
        phones, tones, bert = cleaner.clean_text_bert("m", "原文", "ZH")
        assert phones == ["a", "b"]
        assert bert.shape == (2, 2)

    def test_text_to_sequence_composes(self, monkeypatch):
        def fake_clean(model_name, text, language):
            return "n", ["a", "b"], [1, 2], [1, 1]

        monkeypatch.setattr(cleaner, "clean_text", fake_clean)
        ids = cleaner.text_to_sequence("m", "t", "ZH")
        assert ids[0] == [cleaner._symbol_to_id["a"], cleaner._symbol_to_id["b"]]
        assert ids[1] == [1, 2]
        assert ids[2] == [0, 0]