"""Unit tests for ``mark.py`` — writing transcription sidecars."""

from types import SimpleNamespace

import pytest

from BookerBV2Tool import mark


@pytest.fixture
def fake_sencevoice(monkeypatch):
    """Replace the heavy SenseVoice pipeline with a scripted transcript."""
    def _install(results):
        monkeypatch.setattr(mark, "sencevoice", lambda args: results)
    return _install


class TestMarkFile:
    def test_writes_joined_text(self, tmp_path, fake_sencevoice):
        audio = tmp_path / "seg.wav"
        audio.write_bytes(b"RIFF")
        fake_sencevoice([{"text": "你好"}, {"text": "世界"}])
        mark.mark_file(SimpleNamespace(audio=str(audio)))
        assert (tmp_path / "seg.wav.txt").read_text(encoding="utf-8") == "你好 世界"

    def test_empty_result_writes_empty(self, tmp_path, fake_sencevoice):
        audio = tmp_path / "s.wav"
        audio.write_bytes(b"RIFF")
        fake_sencevoice([])
        mark.mark_file(SimpleNamespace(audio=str(audio)))
        assert (tmp_path / "s.wav.txt").read_text(encoding="utf-8") == ""


class TestMarkFileSafe:
    def test_swallows_exceptions(self, tmp_path, monkeypatch, capsys):
        def boom(args):
            raise RuntimeError("vad failed")

        monkeypatch.setattr(mark, "sencevoice", boom)
        audio = tmp_path / "x.wav"
        audio.write_bytes(b"RIFF")
        mark.mark_file_safe(SimpleNamespace(audio=str(audio)))
        assert not (tmp_path / "x.wav.txt").exists()


class TestMarkDispatch:
    def test_handle_file(self, tmp_path, fake_sencevoice):
        audio = tmp_path / "f.wav"
        audio.write_bytes(b"RIFF")
        fake_sencevoice([{"text": "hi"}])
        mark.mark_handle(SimpleNamespace(audio=str(audio)))
        assert (tmp_path / "f.wav.txt").exists()

    def test_handle_dir(self, tmp_path, fake_sencevoice):
        d = tmp_path / "in"
        d.mkdir()
        for n in ("a.wav", "b.wav"):
            (d / n).write_bytes(b"RIFF")
        fake_sencevoice([{"text": "x"}])
        mark.mark_handle(SimpleNamespace(audio=str(d)))
        assert (d / "a.wav.txt").exists()
        assert (d / "b.wav.txt").exists()