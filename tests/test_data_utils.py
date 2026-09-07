"""Unit tests for ``data_utils.py`` (dataset, collate, bucketed sampler).

``data_utils.py`` uses legacy absolute imports (``import commons``,
``from config import config``, ``from tools.log import logger``,
``from text import ...``).  The ``install_top_level_aliases()`` helper maps those
names onto the real ``BookerBV2Tool`` submodules (plus a tiny ``config``/``tools``
stub) so the module can be imported for testing.
"""

from types import SimpleNamespace

import pytest
import torch

from _compat import install_top_level_aliases

data_utils = install_top_level_aliases()


def _hparams(**overrides):
    base = dict(
        max_wav_value=32768,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        spk2id={"A": 0, "B": 1},
        add_blank=True,
        min_text_len=1,
        max_text_len=10,
        use_mel_posterior_encoder=False,
        cleaned_text=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestTextAudioSpeakerCollate:
    def _sample(self, t, sid=0):
        return (
            torch.arange(t, dtype=torch.long),
            torch.rand(80, t * 3),
            torch.rand(1, t * 100),
            torch.LongTensor([sid]),
            torch.arange(t, dtype=torch.long),
            torch.zeros(t, dtype=torch.long),
            torch.rand(1024, t),
            torch.rand(1024, t),
            torch.rand(1024, t),
        )

    def test_pads_and_sorts_by_spec_length(self):
        batch = [self._sample(3), self._sample(5, sid=1), self._sample(4)]
        collate = data_utils.TextAudioSpeakerCollate()
        (
            text_padded, text_lengths, spec_padded, spec_lengths,
            wav_padded, wav_lengths, sid, tone_padded, language_padded,
            bert_padded, ja_bert_padded, en_bert_padded,
        ) = collate(batch)

        # sorted by spec time descending: 5 -> 4 -> 3
        assert text_lengths.tolist() == [5, 4, 3]
        assert spec_lengths.tolist() == [15, 12, 9]
        assert wav_lengths.tolist() == [500, 400, 300]

        assert text_padded.shape == (3, 5)
        assert spec_padded.shape == (3, 80, 15)
        assert wav_padded.shape == (3, 1, 500)
        assert bert_padded.shape == (3, 1024, 5)
        assert tone_padded.shape == (3, 5)
        assert language_padded.shape == (3, 5)
        assert sid.tolist() == [1, 0, 0]

        # row 0 holds the longest sample (t=5): text [0..4]
        assert text_padded[0, :5].tolist() == [0, 1, 2, 3, 4]
        # row 2 (shortest, t=3) is zero-padded to max length
        assert text_padded[2, :3].tolist() == [0, 1, 2]
        assert text_padded[2, 3:].tolist() == [0, 0]
        # padding is zeroed for shorter samples
        assert spec_padded[2, :, 9:].abs().sum().item() == 0.0

    def test_single_sample(self):
        batch = [self._sample(2)]
        out = data_utils.TextAudioSpeakerCollate()(batch)
        assert out[1].tolist() == [2]
        assert out[0].shape == (1, 2)


class _FakeDataset:
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)


class TestDistributedBucketSampler:
    def test_bucket_assignment(self):
        ds = _FakeDataset([15, 20, 30, 45, 55, 70])
        s = data_utils.DistributedBucketSampler(
            ds, batch_size=2, boundaries=[10, 40, 80], num_replicas=1, rank=0
        )
        # (10, 40] and (40, 80]
        assert s.buckets == [[0, 1, 2], [3, 4, 5]]
        assert s.num_samples_per_bucket == [4, 4]
        assert s.total_size == 8
        assert s.num_samples == 8
        assert len(s) == 4

    def test_iteration_yields_full_batches(self):
        ds = _FakeDataset([15, 20, 30, 45, 55, 70])
        s = data_utils.DistributedBucketSampler(
            ds, batch_size=2, boundaries=[10, 40, 80], num_replicas=1, rank=0, shuffle=True
        )
        s.epoch = 0
        batches = list(s)
        assert len(batches) == 4
        seen = set()
        for batch in batches:
            assert len(batch) == 2
            for idx in batch:
                seen.add(idx)
                bucket = s._bisect(ds.lengths[idx])
                # all members of a batch come from the same bucket
                assert bucket == s._bisect(ds.lengths[batch[0]])
        assert seen == {0, 1, 2, 3, 4, 5}

    def test_deterministic_with_same_epoch(self):
        ds = _FakeDataset([15, 20, 30, 45, 55, 70])
        make = lambda: data_utils.DistributedBucketSampler(
            ds, batch_size=2, boundaries=[10, 40, 80], num_replicas=1, rank=0
        )
        a, b = make(), make()
        a.epoch = 3
        b.epoch = 3
        assert [list(x) for x in a] == [list(x) for x in b]

    def test_bisect(self):
        ds = _FakeDataset([15, 20, 30, 45, 55, 70])
        s = data_utils.DistributedBucketSampler(
            ds, batch_size=2, boundaries=[10, 40, 80], num_replicas=1, rank=0
        )
        assert s._bisect(5) == -1
        assert s._bisect(20) == 0
        assert s._bisect(50) == 1
        assert s._bisect(90) == -1

    def test_empty_bucket_is_pruned(self):
        ds = _FakeDataset([15, 20])
        s = data_utils.DistributedBucketSampler(
            ds, batch_size=2, boundaries=[10, 40, 80], num_replicas=1, rank=0
        )
        assert s.buckets == [[0, 1]]
        assert list(s) == [[0, 1]]


class TestTextAudioSpeakerLoader:
    @pytest.fixture
    def loader(self, tmp_path):
        for name in ("ok1.wav", "ok2.wav", "long.wav"):
            (tmp_path / name).write_bytes(b"0" * 512)  # 512 bytes -> length 1
        list_file = tmp_path / "list.txt"
        list_file.write_text(
            f"{tmp_path / 'ok1.wav'}|A|ZH|t1|n i h ao|1 2 3 4|1 1 1 1\n"
            f"{tmp_path / 'ok2.wav'}|B|ZH|t2|a b c d e|1 2 3 4 5|1 1 1 1 1\n"
            f"{tmp_path / 'long.wav'}|A|ZH|t3|a b c d e f g h i j k l m n o|{'1 ' * 15}|{'1 ' * 15}\n",
            encoding="utf-8",
        )
        loader = data_utils.TextAudioSpeakerLoader(str(list_file), _hparams())
        return loader, tmp_path

    def test_len_and_filtering(self, loader):
        loader, _ = loader
        # 2 rows kept, the 15-phone row is filtered by max_text_len=10
        assert len(loader) == 2
        assert loader.lengths == [1, 1]

    def test_get_sid(self, loader):
        loader, _ = loader
        sid = loader.get_sid("1")
        assert sid.tolist() == [1]

    def test_get_audio_computes_spectrogram(self, loader, monkeypatch):
        loader, _ = loader
        audio = torch.randn(22050) * 0.5
        monkeypatch.setattr(data_utils, "load_wav_to_torch", lambda fn: (audio, 22050))
        spec, audio_norm = loader.get_audio("dummy.wav")
        # [n_fft//2 + 1, frames] after squeeze(0)
        assert spec.shape[0] == 513
        assert audio_norm.shape == (1, 22050)

    def test_get_audio_uses_cached_spec(self, loader, tmp_path, monkeypatch):
        loader, _ = loader
        audio = torch.randn(8000) * 0.5
        monkeypatch.setattr(data_utils, "load_wav_to_torch", lambda fn: (audio, 22050))
        cached = torch.rand(513, 20)
        wav = tmp_path / "cached.wav"
        wav.write_bytes(b"x")
        torch.save(cached, str(tmp_path / "cached.spec.pt"))
        spec, _ = loader.get_audio(str(wav))
        torch.testing.assert_close(spec, cached)

    def test_get_text_intersperses_and_aligns(self, loader, monkeypatch):
        loader, _ = loader
        bert_len = 2 * 4 + 1  # 4 phones interspersed with blank
        monkeypatch.setattr(
            data_utils.torch, "load", lambda p: torch.randn(1024, bert_len)
        )
        word2ph = [1, 1, 1, 1]
        bert, ja_bert, en_bert, phone, tone, language = loader.get_text(
            "t", word2ph, ["n", "i", "h", "ao"], [1, 2, 3, 4], "ZH", "x.wav"
        )
        assert phone.shape == (bert_len,)
        assert tone.shape == (bert_len,)
        assert language.shape == (bert_len,)
        # word2ph is mutated in place: each doubled, first +1
        assert word2ph == [3, 2, 2, 2]
        assert bert.shape == (1024, bert_len)
        assert ja_bert.shape == (1024, bert_len)
        assert en_bert.shape == (1024, bert_len)