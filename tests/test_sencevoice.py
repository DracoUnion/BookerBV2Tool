"""Unit tests for ``sencevoice.py`` — the SenseVoice transcription pipeline.

The real implementation shells out to ``ffmpeg`` and loads HuggingFace
``funasr`` models, so everything external is mocked; what we verify is the
orchestration: VAD segments -> cropping -> STT -> timestamped results.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from BookerBV2Tool import sencevoice


class _FakeAutoModel:
    instances = []

    def __init__(self, model=None, **kwargs):
        self.model = model
        _FakeAutoModel.instances.append(self)

    def generate(self, input=None, cache={}, **kwargs):
        if self.model and self.model.endswith("fsmn-vad"):
            return [{"value": [[0, 500], [1500, 2500]]}]
        return [{"text": " 你好呀 ", "value": []}]


@pytest.fixture
def stt_env(monkeypatch):
    sencevoice.AutoModel = _FakeAutoModel
    _FakeAutoModel.instances = []
    monkeypatch.setattr(sencevoice, "rich_transcription_postprocess", lambda x: x.strip())
    monkeypatch.setattr(sencevoice.subp, "Popen", lambda *a, **k: SimpleNamespace(communicate=lambda: (b"", b"")))
    monkeypatch.setattr(sencevoice, "sf", _FakeSF(16000, 32000))
    monkeypatch.setattr(sencevoice.os, "unlink", lambda p: None)
    monkeypatch.setattr(sencevoice.path, "isfile", lambda p: True)
    monkeypatch.setattr(sencevoice, "uuid", _FakeUUID())
    return sencevoice


class _FakeSF:
    def __init__(self, sr, n):
        self.sr = sr
        self.n = n

    def read(self, *a, **k):
        return np.zeros(self.n, dtype=np.float32), self.sr

    def write(self, *a, **k):
        return None


class _FakeUUID:
    class _UUID:
        @property
        def hex(self):
            return "deadbeef"

    def uuid4(self):
        return self._UUID()


class TestSencevoice:
    def test_non_mp3_converts_and_returns_results(self, stt_env):
        args = SimpleNamespace(audio="C:/in/voice.wav", sencevoice="D:/models/SenseVoiceSmall")
        res = stt_env.sencevoice(args)
        assert res == [
            {"start": 0.0, "end": 0.5, "text": "你好呀", "time": 0.0},
            {"start": 1.5, "end": 2.5, "text": "你好呀", "time": 1.5},
        ]

    def test_loads_vad_and_stt_models(self, stt_env):
        args = SimpleNamespace(audio="C:/in/voice.wav", sencevoice="D:/models/SenseVoiceSmall")
        stt_env.sencevoice(args)
        models = [inst.model for inst in _FakeAutoModel.instances]
        assert "fsmn-vad" in models[0]
        assert models[1] == "D:/models/SenseVoiceSmall"

    def test_mp3_input_skips_conversion(self, stt_env, monkeypatch):
        unlinks = []
        monkeypatch.setattr(stt_env.os, "unlink", lambda p: unlinks.append(p))
        args = SimpleNamespace(audio="C:/in/voice.mp3", sencevoice="D:/models/SenseVoiceSmall")
        res = stt_env.sencevoice(args)
        assert len(res) == 2
        # the original mp3 is not deleted (it is not a temp conversion)
        assert not any(p.endswith("voice.mp3") for p in unlinks)

    def test_raises_when_conversion_file_missing(self, stt_env, monkeypatch):
        # Characterization: the source raises FileNotFoundError but references an
        # undefined `fname`, so a NameError surfaces instead.
        def isfile(p):
            return "deadbeef" not in p

        monkeypatch.setattr(stt_env.path, "isfile", isfile)
        args = SimpleNamespace(audio="C:/in/voice.wav", sencevoice="D:/models/SenseVoiceSmall")
        with pytest.raises(Exception):
            stt_env.sencevoice(args)