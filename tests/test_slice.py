"""Unit tests for ``BookerBV2Tool/slice.py`` — the VAD-style Slicer."""

import numpy as np
import pytest
from types import SimpleNamespace

from BookerBV2Tool.slice import Slicer, get_rms, slice_handle

SR = 16000


def _concat(chunks):
    if len(chunks[0].shape) > 1:
        return np.concatenate([c for c in chunks], axis=1)
    return np.concatenate(chunks)


class TestGetRms:
    def test_shape(self):
        y = np.zeros(16000, dtype=np.float32)
        out = get_rms(y, frame_length=2048, hop_length=512)
        # For a 1-D signal: [1, n_frames] — squeeze(0) gives the Slicer's frames.
        assert out.ndim == 2
        assert out.shape[0] == 1
        assert out.squeeze(0).ndim == 1

    def test_silence_is_zero(self):
        y = np.zeros(16000, dtype=np.float32)
        out = get_rms(y, frame_length=1024, hop_length=512)
        assert np.allclose(out, 0.0)

    def test_constant_gain(self):
        y = np.full(16000, 0.5, dtype=np.float32)
        out = get_rms(y, frame_length=1024, hop_length=512)
        # boundary frames see the zero padding; interior frames are ~0.5
        assert out.shape[0] == 1
        assert np.allclose(out[0, 1:-1], 0.5, atol=1e-2)


class TestSlicerInit:
    def test_valid(self):
        Slicer(sr=16000, min_length=2000, min_interval=300, hop_size=20)

    def test_raises_when_min_length_lt_min_interval(self):
        with pytest.raises(ValueError):
            Slicer(sr=16000, min_length=100, min_interval=300, hop_size=20)

    def test_raises_when_max_sil_kept_lt_hop(self):
        with pytest.raises(ValueError):
            Slicer(sr=16000, min_length=2000, min_interval=300, hop_size=20, max_sil_kept=10)

    def test_derived_attributes(self):
        s = Slicer(sr=16000, threshold=-40, min_length=2000, min_interval=300, hop_size=20, max_sil_kept=500)
        assert s.threshold == pytest.approx(10 ** (-40 / 20))
        assert s.hop_size == 320  # 16000 * 20 / 1000
        assert s.min_length == 100  # frames
        assert s.min_interval == 15
        assert s.max_sil_kept == 25


class TestSlicerSlice:
    @pytest.fixture
    def slicer(self):
        return Slicer(
            sr=SR,
            threshold=-40,
            min_length=1000,  # 50 frames
            min_interval=300,  # 15 frames
            hop_size=20,
            max_sil_kept=500,  # 25 frames
        )

    @staticmethod
    def _three_part(silence_sample):
        tone, silence, _ = silence_sample
        one_sec = len(tone)
        return np.concatenate([tone, silence, silence, tone])  # 1s + 2s + 1s

    def test_short_audio_returns_unchanged(self, slicer):
        wav = np.full(SR // 10, 0.5, dtype=np.float32)  # 100 ms
        assert slicer.slice(wav) == [wav]

    def test_fully_loud_signal_is_not_split(self, slicer, silence_sample):
        tone, _, _ = silence_sample
        wav = np.tile(tone, 3)
        chunks = slicer.slice(wav)
        assert len(chunks) == 1

    def test_silence_gap_is_split(self, slicer, silence_sample):
        wav = self._three_part(silence_sample)
        chunks = slicer.slice(wav)
        assert len(chunks) == 2
        # The long interior silence must be removed, so the concatenation is
        # strictly shorter than the input.
        assert _concat(chunks).shape[0] < wav.shape[0]

    def test_no_empty_chunks(self, slicer, silence_sample):
        wav = self._three_part(silence_sample)
        chunks = slicer.slice(wav)
        assert all(c.shape[0] > 0 for c in chunks)

    def test_stereo_is_downmixed_for_detection(self, slicer, silence_sample):
        tone, silence, _ = silence_sample
        mono = np.concatenate([tone, silence, silence, tone])
        stereo = np.stack([mono, mono * 0.5])  # [2, n]
        chunks = slicer.slice(stereo)
        assert len(chunks) == 2

    def test_ambiguity_handling_returns_list(self, silence_sample):
        tone, silence, _ = silence_sample
        wav = np.concatenate([tone * 0.001, silence, tone * 0.001])
        s = Slicer(sr=SR, min_length=100, min_interval=50, hop_size=20, max_sil_kept=100)
        chunks = s.slice(wav)
        assert isinstance(chunks, list)


class TestApplySlice:
    def test_clip_at_end(self):
        s = Slicer(sr=SR, min_length=1000, min_interval=300, hop_size=20)
        wav = np.full(2000, 0.5, dtype=np.float32)
        out = s._apply_slice(wav, 1, 1000)
        assert out.shape[0] == min(2000, 1000 * s.hop_size) - 1 * s.hop_size

    def test_stereo(self):
        s = Slicer(sr=SR, min_length=1000, min_interval=300, hop_size=20)
        wav = np.ones((2, 3000), dtype=np.float32)
        out = s._apply_slice(wav, 1, 5)
        assert out.shape == (2, 4 * s.hop_size)


class TestSliceHandle:
    def test_writes_chunks(self, tmp_path, silence_sample, monkeypatch):
        import soundfile as sf

        tone, silence, _ = silence_sample
        wav = np.concatenate([tone, silence, silence, tone, silence[: SR // 2]])
        audio_path = tmp_path / "input.wav"
        sf.write(str(audio_path), wav, SR)
        out_dir = tmp_path / "out"

        args = SimpleNamespace(
            audio=str(audio_path),
            out=str(out_dir),
            db_thresh=-40,
            min_length=1000,
            min_interval=300,
            hop_size=20,
            max_sil_kept=500,
        )
        slice_handle(args)

        files = sorted(out_dir.glob("input_*.wav"))
        assert len(files) >= 1
        for f in files:
            data, sr = sf.read(str(f))
            assert sr == SR
            assert len(data) > 0

    def test_out_defaults_to_audio_dir(self, tmp_path, silence_sample):
        import soundfile as sf

        tone, silence, _ = silence_sample
        wav = np.concatenate([tone, silence, silence, tone])
        audio = tmp_path / "dxc" / "input.wav"
        audio.parent.mkdir()
        sf.write(str(audio), wav, SR)
        args = SimpleNamespace(
            audio=str(audio),
            out=None,
            db_thresh=-40,
            min_length=1000,
            min_interval=300,
            hop_size=20,
            max_sil_kept=500,
        )
        slice_handle(args)
        assert list(audio.parent.glob("input_*.wav"))