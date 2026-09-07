"""Unit tests for ``mklist.py`` — building ``speaker_list.json``."""

import json
from types import SimpleNamespace

from BookerBV2Tool import mklist


def _mk(args, dirpath, files):
    dirpath.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        p = dirpath / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))


class TestMklistHandle:
    def test_builds_speaker_list(self, tmp_path, capsys):
        _mk(None, tmp_path, {
            "a.wav": b"",
            "a.wav.txt": "你好",
            "b.wav": b"",
            "b.wav.txt": "世界",
        })
        mklist.mklist_handle(
            SimpleNamespace(dir=str(tmp_path), role="wizard", lang="ZH")
        )
        data = json.loads((tmp_path / "speaker_list.json").read_text(encoding="utf-8"))
        assert data == [
            {"file": "a.wav", "role": "wizard", "lang": "ZH", "sub": "你好"},
            {"file": "b.wav", "role": "wizard", "lang": "ZH", "sub": "世界"},
        ]

    def test_skips_missing_and_empty_sidecars(self, tmp_path, capsys):
        _mk(None, tmp_path, {
            "ok.wav": b"",
            "ok.wav.txt": "有内容",
            "empty.wav": b"",
            "empty.wav.txt": "",
            "nosub.wav": b"",
        })
        mklist.mklist_handle(
            SimpleNamespace(dir=str(tmp_path), role="wizard", lang="ZH")
        )
        data = json.loads((tmp_path / "speaker_list.json").read_text(encoding="utf-8"))
        assert [d["file"] for d in data] == ["ok.wav"]
        out = capsys.readouterr().out
        assert "不存在" in out and "为空" in out

    def test_no_wavs_returns_early(self, tmp_path, capsys):
        (tmp_path / "readme.txt").write_text("hi", encoding="utf-8")
        mklist.mklist_handle(
            SimpleNamespace(dir=str(tmp_path), role="r", lang="ZH")
        )
        assert not (tmp_path / "speaker_list.json").exists()
        assert "WAV" in capsys.readouterr().out

    def test_dir_must_exist(self, tmp_path, capsys):
        mklist.mklist_handle(
            SimpleNamespace(dir=str(tmp_path / "nope"), role="r", lang="ZH")
        )
        assert not (tmp_path / "nope" / "speaker_list.json").exists()

    def test_empty_result_prints_warning(self, tmp_path, capsys):
        _mk(None, tmp_path, {"a.wav": b"", "a.wav.txt": ""})
        mklist.mklist_handle(
            SimpleNamespace(dir=str(tmp_path), role="r", lang="ZH")
        )
        assert "未找到任何标注数据" in capsys.readouterr().out