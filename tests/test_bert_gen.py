"""Unit tests for ``bert_gen.py`` — generating BERT phone embeddings."""

from types import SimpleNamespace

import torch
import pytest

from BookerBV2Tool import bert_gen


def _line(tmp_path, phones=("n", "i")):
    return {
        "phones": list(phones),
        "tones": [1] * len(phones),
        "lang": "ZH",
        "word2ph": [1] * len(phones),
        "file": str(tmp_path / "utt.wav"),
        "norm_sub": "ni",
    }


def _args():
    return SimpleNamespace(chinese_bert="cb", enlish_bert="eb", japanese_bert="jb")


def _bert_width(phones, add_blank):
    from BookerBV2Tool.text import cleaner

    ids = cleaner.cleaned_text_to_sequence(list(phones), [1] * len(phones), "ZH")[0]
    if add_blank:
        from BookerBV2Tool import commons

        return len(commons.intersperse(ids, 0))
    return len(ids)


class TestProcessLine:
    def test_writes_bert_with_blank(self, tmp_path, monkeypatch):
        (tmp_path / "utt.wav").write_bytes(b"RIFF")
        line = _line(tmp_path)
        monkeypatch.setattr(bert_gen, "get_bert_feature", lambda *a, **k: torch.zeros(1024, 5))
        bert_gen.process_line(line, add_blank=True, args=_args())
        saved = torch.load(str(tmp_path / "utt_bert.pt"), map_location="cpu")
        assert saved.shape == (1024, 5)

    def test_writes_bert_without_blank(self, tmp_path, monkeypatch):
        (tmp_path / "utt.wav").write_bytes(b"RIFF")
        line = _line(tmp_path)
        monkeypatch.setattr(bert_gen, "get_bert_feature", lambda *a, **k: torch.zeros(1024, 2))
        bert_gen.process_line(line, add_blank=False, args=_args())
        saved = torch.load(str(tmp_path / "utt_bert.pt"), map_location="cpu")
        assert saved.shape == (1024, 2)

    def test_skips_when_bert_exists(self, tmp_path, monkeypatch):
        (tmp_path / "utt.wav").write_bytes(b"RIFF")
        (tmp_path / "utt_bert.pt").write_bytes(b"existing")
        line = _line(tmp_path)
        called = []
        monkeypatch.setattr(
            bert_gen, "get_bert_feature", lambda *a, **k: called.append(1)
        )
        bert_gen.process_line(line, add_blank=True, args=_args())
        assert called == []

    def test_safe_swallows_errors(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "utt.wav").write_bytes(b"RIFF")
        line = _line(tmp_path)
        monkeypatch.setattr(bert_gen, "get_bert_feature", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
        bert_gen.process_line_safe(line, True, _args())  # must not raise


class TestBertGenHandle:
    def test_writes_all_pending(self, tmp_path, monkeypatch, capsys):
        config = {
            "data": {
                "training_files": str(tmp_path / "train.json"),
                "validation_files": str(tmp_path / "val.json"),
                "add_blank": True,
            }
        }
        (tmp_path / "config.json").write_text(__import__("json").dumps(config), encoding="utf-8")
        lines = [
            {"phones": ["n"], "tones": [1], "lang": "ZH", "word2ph": [1], "file": str(tmp_path / "a.wav"), "norm_sub": "a"},
            {"phones": ["i"], "tones": [1], "lang": "ZH", "word2ph": [1], "file": str(tmp_path / "b.wav"), "norm_sub": "i"},
        ]
        (tmp_path / "train.json").write_text(__import__("json").dumps(lines), encoding="utf-8")
        (tmp_path / "val.json").write_text(__import__("json").dumps([]), encoding="utf-8")
        for n in ("a", "b"):
            (tmp_path / f"{n}.wav").write_bytes(b"RIFF")
        monkeypatch.setattr(bert_gen, "get_bert_feature", lambda *a, **k: torch.zeros(1024, 3))
        monkeypatch.setattr(bert_gen, "ProcessPoolExecutor", lambda *a, **k: _InlinePool())

        bert_gen.bert_gen_handle(SimpleNamespace(config=str(tmp_path / "config.json"), num_processes=1, chinese_bert="cb", enlish_bert="eb", japanese_bert="jb"))
        assert (tmp_path / "a_bert.pt").exists()
        assert (tmp_path / "b_bert.pt").exists()

    def test_no_data_returns(self, tmp_path, monkeypatch, capsys):
        config = {"data": {"training_files": str(tmp_path / "t.json"), "validation_files": str(tmp_path / "v.json"), "add_blank": True}}
        (tmp_path / "config.json").write_text(__import__("json").dumps(config), encoding="utf-8")
        (tmp_path / "t.json").write_text("[]", encoding="utf-8")
        (tmp_path / "v.json").write_text("[]", encoding="utf-8")
        monkeypatch.setattr(bert_gen, "ProcessPoolExecutor", lambda *a, **k: _InlinePool())
        bert_gen.bert_gen_handle(SimpleNamespace(config=str(tmp_path / "config.json"), num_processes=1, chinese_bert="cb", enlish_bert="eb", japanese_bert="jb"))
        assert "未找到训练或测试文件" in capsys.readouterr().out


class _Future:
    def result(self):
        return None


class _InlinePool:
    """Run submitted calls synchronously in-process (no subprocess)."""
    def submit(self, fn, *a, **k):
        fn(*a, **k)
        return _Future()
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False