"""Unit tests for ``BookerBV2Tool/resample.py``."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from BookerBV2Tool import resample


def _to_wav(path, sr=22050, n=44100):
    data = (np.linspace(-1, 1, n) * 0.5).astype(np.float32)
    sf.write(str(path), data, sr)


class TestResampleFile:
    def test_resamples_to_target_sr(self, tmp_path):
        src = tmp_path / "src.wav"
        out = tmp_path / "dst"
        _to_wav(src, sr=22050, n=44100)
        resample.resample_file(
            SimpleNamespace(audio=str(src), out=str(out), sr=16000)
        )
        dst = out / "src.wav"
        assert dst.exists()
        info = sf.info(str(dst))
        assert info.samplerate == 16000

    def test_ignores_non_wav(self, tmp_path, capsys):
        src = tmp_path / "clip.mp3"
        src.write_bytes(b"not really mp3")
        out = tmp_path / "dst"
        resample.resample_file(
            SimpleNamespace(audio=str(src), out=str(out), sr=16000)
        )
        assert not out.exists()
        assert "WAV" in capsys.readouterr().out

    def test_safe_swallows_errors(self, tmp_path, capsys):
        missing = tmp_path / "missing.wav"
        resample.resample_file_safe(
            SimpleNamespace(audio=str(missing), out=str(tmp_path / "o"), sr=16000)
        )
        # no exception should escape; traceback printed to stderr
        assert missing.exists() is False


class TestResampleHandle:
    def test_dispatch_file(self, tmp_path):
        src = tmp_path / "a.wav"
        out = tmp_path / "o"
        _to_wav(src, sr=22050, n=8000)
        resample.resample_handle(
            SimpleNamespace(audio=str(src), out=str(out), sr=22050, threads=2)
        )
        assert out.joinpath("a.wav").exists()

    def test_dispatch_dir(self, tmp_path):
        d = tmp_path / "in"
        d.mkdir()
        out = tmp_path / "out"
        for i in range(3):
            _to_wav(d / f"f{i}.wav", sr=22050, n=4000)
        resample.resample_handle(
            SimpleNamespace(audio=str(d), out=str(out), sr=11025, threads=1)
        )
        names = {p.name for p in out.glob("*.wav")}
        assert names == {"f0.wav", "f1.wav", "f2.wav"}
        assert all(sf.info(str(out / n)).samplerate == 11025 for n in names)