"""Shared pytest fixtures and environment setup for the BookerBV2Tool test suite.

The repository's source modules live in ``BookerBV2Tool/`` but a few of them
(e.g. ``data_utils.py``) use legacy absolute imports (``import commons``,
``from config import config``) that assume the package directory is on
``sys.path``.  This file makes those imports resolvable so the tests can
import the real modules.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "BookerBV2Tool"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))


@pytest.fixture
def make_wav(tmp_path):
    """Factory fixture that writes a small mono int16 ``.wav`` and returns its path.

    ``data=None`` generates ``sr`` samples of uniform noise scaled to ~50% of
    full scale (deterministic via a fixed RNG seed).
    """

    def _write(name="audio.wav", data=None, sr=22050):
        import numpy as np
        from scipy.io import wavfile

        if data is None:
            data = (np.random.default_rng(0).uniform(-1, 1, sr) * 16384).astype(
                np.int16
            )
        path = tmp_path / name
        wavfile.write(str(path), sr, data)
        return path

    return _write


@pytest.fixture
def silence_sample():
    """Return ``(tone_samples, silence_samples, sr)`` for building synthetic audio.

    ``tone_samples`` is a 0.5-amplitude 220 Hz sine wave; ``silence_samples`` is
    the same number of zero samples.  Useful for the ``Slicer`` tests.
    """
    import numpy as np

    sr = 16000
    n = sr  # 1 second
    t = np.arange(n) / sr
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    silence = np.zeros(n, dtype=np.float32)
    return tone, silence, sr