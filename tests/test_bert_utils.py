"""Unit tests for ``text/bert_utils.py`` — model file downloading."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from BookerBV2Tool.text import bert_utils


class TestCheckBert:
    def test_all_files_present_no_download(self, tmp_path, monkeypatch):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "pytorch_model.bin").write_bytes(b"\x00")
        hub = MagicMock()
        monkeypatch.setattr(bert_utils, "hf_hub_download", hub)
        bert_utils._check_bert("org/model", ["config.json", "pytorch_model.bin"], str(tmp_path))
        hub.assert_not_called()

    def test_missing_file_triggers_hf_download(self, tmp_path, monkeypatch):
        hub = MagicMock()
        monkeypatch.setattr(bert_utils, "hf_hub_download", hub)
        bert_utils._check_bert("org/model", ["config.json", "pytorch_model.bin"], str(tmp_path))
        assert hub.call_count == 2
        # hf_hub_download is called positionally with (repo_id, filename, ...)
        assert hub.call_args.args[0] == "org/model"
        assert hub.call_args.kwargs["local_dir_use_symlinks"] is False

    def test_openi_mirror(self, tmp_path, monkeypatch):
        openi = ModuleType("openi")
        openi.model = MagicMock()
        sys.modules["openi"] = openi
        monkeypatch.setattr(bert_utils, "MIRROR", "openi")
        bert_utils._check_bert("org/model", ["pytorch_model.bin"], str(tmp_path))
        openi.model.download_model.assert_called_once()
        sys.modules.pop("openi", None)