"""Unit tests for ``preproc.py`` — dataset train/validation split + config."""

import json
from types import SimpleNamespace

from BookerBV2Tool import preproc


def _fake_clean_text(model_name, text, lang):
    return "NORM", ["n", "i"], [1, 2], [1, 1]


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


class TestPreprocessHandle:
    def _args(self, tmp_path):
        return SimpleNamespace(
            transcription_path=str(tmp_path / "transcription.json"),
            train_path=str(tmp_path / "train_list.json"),
            val_path=str(tmp_path / "val_list.json"),
            config_path=str(tmp_path / "config.json"),
            val_per_lang=1,
            max_val_total=3,
            chinese_bert="cb",
            enlish_bert="eb",
            japanese_bert="jb",
        )

    def _make_wavs(self, tmp_path):
        for name in ("a.wav", "b.wav", "c.wav"):
            (tmp_path / name).write_bytes(b"RIFFxxxx")

    def test_splits_roles_and_writes_config(self, tmp_path, monkeypatch, capsys):
        self._make_wavs(tmp_path)
        wav_a = str(tmp_path / "a.wav")
        wav_b = str(tmp_path / "b.wav")
        wav_c = str(tmp_path / "c.wav")
        lines = [
            {"file": wav_a, "role": "wizard", "lang": "ZH", "sub": "你好"},
            {"file": wav_b, "role": "wizard", "lang": "ZH", "sub": "世界"},
            {"file": wav_c, "role": "other", "lang": "EN", "sub": "hello"},
        ]
        _write_json(tmp_path / "transcription.json", lines)

        raw_cfg = {
            "data": {"spk2id": {}, "n_speakers": 0, "training_files": "", "validation_files": ""}
        }
        raw_path = tmp_path / "raw.json"
        _write_json(raw_path, raw_cfg)

        monkeypatch.setattr(preproc, "clean_text", _fake_clean_text)
        monkeypatch.setattr(preproc, "RAW_CONFIG_PATH", str(raw_path))

        args = self._args(tmp_path)
        preproc.preprocess_handle(args)

        # cleaned cache created
        assert (tmp_path / "transcription_cleaned.json").exists()
        cleaned = json.loads((tmp_path / "transcription_cleaned.json").read_text(encoding="utf-8"))
        assert len(cleaned) == 3
        assert all("phones" in c and "norm_sub" in c for c in cleaned)

        train = json.loads((tmp_path / "train_list.json").read_text(encoding="utf-8"))
        val = json.loads((tmp_path / "val_list.json").read_text(encoding="utf-8"))
        # every valid line ends up in exactly one split
        assert len(train) + len(val) == 3
        assert len(val) == 2  # one per role (2 roles)

        cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert cfg["version"] == "2.3"
        assert cfg["data"]["spk2id"] == {"wizard": 0, "other": 1}
        assert cfg["data"]["n_speakers"] == 2
        assert cfg["data"]["training_files"].endswith("train_list.json")
        assert cfg["data"]["validation_files"].endswith("val_list.json")

    def test_filters_duplicates_and_missing(self, tmp_path, monkeypatch, capsys):
        self._make_wavs(tmp_path)
        wav_a = str(tmp_path / "a.wav")
        lines = [
            {"file": wav_a, "role": "wizard", "lang": "ZH", "sub": "第一次"},
            {"file": wav_a, "role": "wizard", "lang": "ZH", "sub": "重复"},
            {"file": str(tmp_path / "ghost.wav"), "role": "wizard", "lang": "ZH", "sub": "不存在"},
        ]
        _write_json(tmp_path / "transcription.json", lines)
        raw_path = tmp_path / "raw.json"
        _write_json(raw_path, {"data": {"spk2id": {}, "n_speakers": 0}})
        monkeypatch.setattr(preproc, "clean_text", _fake_clean_text)
        monkeypatch.setattr(preproc, "RAW_CONFIG_PATH", str(raw_path))

        preproc.preprocess_handle(self._args(tmp_path))
        out = capsys.readouterr().out
        assert "重复音频文本" in out
        assert "没有找到对应的音频" in out
        train = json.loads((tmp_path / "train_list.json").read_text(encoding="utf-8"))
        val = json.loads((tmp_path / "val_list.json").read_text(encoding="utf-8"))
        subs = [c["sub"] for c in train + val]
        assert "重复" not in subs and "不存在" not in subs

    def test_reuses_existing_cleaned_file(self, tmp_path, monkeypatch):
        self._make_wavs(tmp_path)
        wav_a = str(tmp_path / "a.wav")
        lines = [{"file": wav_a, "role": "wizard", "lang": "ZH", "sub": "你好"}]
        _write_json(tmp_path / "transcription.json", lines)
        # pre-create the cleaned cache
        _write_json(
            tmp_path / "transcription_cleaned.json",
            [{"file": wav_a, "role": "wizard", "lang": "ZH", "sub": "你好", "phones": ["n", "i"], "tones": [1, 2], "word2ph": [1, 1], "norm_sub": "N"}],
        )
        raw_path = tmp_path / "raw.json"
        _write_json(raw_path, {"data": {"spk2id": {}, "n_speakers": 0}})
        monkeypatch.setattr(preproc, "RAW_CONFIG_PATH", str(raw_path))
        calls = []
        monkeypatch.setattr(
            preproc, "clean_text",
            lambda *a, **k: (calls.append(a) or _fake_clean_text(*a, **k)),
        )
        preproc.preprocess_handle(self._args(tmp_path))
        assert calls == []  # cleaning skipped

    def test_empty_transcription_returns(self, tmp_path, monkeypatch, capsys):
        _write_json(tmp_path / "transcription.json", [])
        raw_path = tmp_path / "raw.json"
        _write_json(raw_path, {"data": {"spk2id": {}, "n_speakers": 0}})
        monkeypatch.setattr(preproc, "RAW_CONFIG_PATH", str(raw_path))
        preproc.preprocess_handle(self._args(tmp_path))
        assert "为空" in capsys.readouterr().out
        assert not (tmp_path / "train_list.json").exists()