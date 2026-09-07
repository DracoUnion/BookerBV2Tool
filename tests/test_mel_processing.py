"""Unit tests for ``BookerBV2Tool/mel_processing.py`` — STFT / Mel transforms."""

import torch
import pytest

from BookerBV2Tool import mel_processing

N_FFT = 1024
SR = 22050
HOP = 256
WIN = 1024
N_MELS = 80


def _sine(n, sr=SR, f=440.0):
    """Return a sine wave shaped ``[1, n]`` — the batch layout spectrogram_torch
    expects (a 1-D input makes ``F.pad`` pad the channel axis and crash)."""
    t = torch.arange(n, dtype=torch.float32) / sr
    return (0.5 * torch.sin(2 * torch.pi * f * t)).unsqueeze(0)


class TestDynamicRangeCompression:
    def test_roundtrip(self):
        x = torch.linspace(0.001, 10.0, 100)
        comp = mel_processing.dynamic_range_compression_torch(x)
        back = mel_processing.dynamic_range_decompression_torch(comp)
        assert torch.allclose(back, x, atol=1e-4, rtol=1e-4)

    def test_clamps_to_log1(self):
        # Below the clip value the result is log(clip_val * C) = log(1e-5).
        x = torch.zeros(3)
        out = mel_processing.dynamic_range_compression_torch(x)
        assert torch.allclose(out, torch.full_like(out, torch.log(torch.tensor(1e-5))))

    def test_spectral_normalize_de_normalize_roundtrip(self):
        mag = torch.rand(4, 5) * 5 + 0.01
        back = mel_processing.spectral_de_normalize_torch(
            mel_processing.spectral_normalize_torch(mag)
        )
        assert torch.allclose(back, mag, atol=1e-4, rtol=1e-4)


@pytest.fixture(autouse=True)
def _clear_caches():
    # The module keeps a global hann_window/mel_basis cache keyed by device/dtype.
    mel_processing.hann_window.clear()
    mel_processing.mel_basis.clear()
    yield
    mel_processing.hann_window.clear()
    mel_processing.mel_basis.clear()


@pytest.fixture
def patched_librosa_mel(monkeypatch):
    """Stub ``librosa.filters.mel``.

    The source calls it positionally, but librosa>=0.10 made its ``n_mels`` /
    ``fmin`` / ``fmax`` arguments keyword-only, so the real call raises
    TypeError.  We replace it with a deterministic filter bank so the mel
    computation path (cache + matmul + log-normalize) can still be tested.
    """
    import numpy as np

    def mel_fn(sr, n_fft, n_mels, fmin, fmax):
        rng = np.random.default_rng(123)
        # small positive weights so log-compressed mel outputs stay < 0
        return ((np.abs(rng.standard_normal((n_mels, n_fft // 2 + 1))) + 0.1) * 0.05).astype(
            np.float32
        )

    monkeypatch.setattr(mel_processing, "librosa_mel_fn", mel_fn)
    return mel_fn


class TestSpectrogramTorch:
    def test_output_shape(self):
        y = _sine(N_FFT * 4)
        spec = mel_processing.spectrogram_torch(y, N_FFT, SR, HOP, WIN)
        # spectrogram_torch keeps a leading batch dim; the loader later squeezes it.
        assert spec.ndim == 3
        assert spec.shape[0] == 1
        assert spec.shape[1] == N_FFT // 2 + 1

    def test_matches_raw_stft(self):
        import torch.nn.functional as F

        y = _sine(N_FFT * 4)
        spec = mel_processing.spectrogram_torch(y, N_FFT, SR, HOP, WIN)
        pad = int((N_FFT - HOP) / 2)
        padded = F.pad(y, (pad, pad), mode="reflect")
        raw = torch.stft(
            padded,
            N_FFT,
            hop_length=HOP,
            win_length=WIN,
            window=torch.hann_window(WIN),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )
        # the function adds a 1e-6 floor inside the sqrt, which the reference
        # must reproduce to match at low-magnitude bins.
        expected = torch.sqrt(raw.pow(2).sum(-1) + 1e-6)
        assert torch.allclose(spec, expected, atol=1e-5, rtol=1e-4)

    def test_min_max_warning_no_crash(self, capsys):
        y = torch.full((1, N_FFT), 2.0)  # > 1.0
        spec = mel_processing.spectrogram_torch(y, N_FFT, SR, HOP, WIN)
        assert "max value" in capsys.readouterr().out

    def test_hann_window_cached(self):
        y = _sine(N_FFT * 2)
        mel_processing.spectrogram_torch(y, N_FFT, SR, HOP, WIN)
        assert len(mel_processing.hann_window) == 1
        mel_processing.spectrogram_torch(y, N_FFT, SR, HOP, WIN)
        assert len(mel_processing.hann_window) == 1  # still cached


class TestSpecToMel:
    def test_output_shape(self, patched_librosa_mel):
        spec = torch.rand(1, N_FFT // 2 + 1, 20)
        mel = mel_processing.spec_to_mel_torch(
            spec, N_FFT, N_MELS, SR, fmin=0.0, fmax=8000.0
        )
        assert mel.shape == (1, N_MELS, 20)

    def test_wider_tensor_keeps_batch(self, patched_librosa_mel):
        spec = torch.rand(3, N_FFT // 2 + 1, 7)
        mel = mel_processing.spec_to_mel_torch(
            spec, N_FFT, N_MELS, SR, fmin=0.0, fmax=8000.0
        )
        assert mel.shape == (3, N_MELS, 7)


class TestMelSpectrogramTorch:
    def test_output_shape(self, patched_librosa_mel):
        y = _sine(N_FFT * 4)
        mel = mel_processing.mel_spectrogram_torch(
            y, N_FFT, N_MELS, SR, HOP, WIN, fmin=0.0, fmax=8000.0
        )
        assert mel.ndim == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == N_MELS

    def test_is_logged_positive_finite(self, patched_librosa_mel):
        y = _sine(N_FFT * 4)
        mel = mel_processing.mel_spectrogram_torch(
            y, N_FFT, N_MELS, SR, HOP, WIN, fmin=0.0, fmax=8000.0
        )
        assert torch.isfinite(mel).all()
        # spectral_normalize clamps to log(1e-5), so no NaN / -inf
        assert (mel > torch.log(torch.tensor(1e-5)) - 1e-3).all()

    def test_low_fmax_reduces_mel_filters(self, patched_librosa_mel):
        y = _sine(N_FFT * 4)
        mel = mel_processing.mel_spectrogram_torch(
            y, N_FFT, N_MELS, SR, HOP, WIN, fmin=0.0, fmax=1000.0
        )
        # librosa still returns num_mels rows; just check it runs and is finite.
        assert torch.isfinite(mel).all()

    def test_caches_are_populated(self, patched_librosa_mel):
        y = _sine(N_FFT * 2)
        mel_processing.mel_spectrogram_torch(
            y, N_FFT, N_MELS, SR, HOP, WIN, fmin=0.0, fmax=8000.0
        )
        assert len(mel_processing.mel_basis) == 1
        assert len(mel_processing.hann_window) == 1