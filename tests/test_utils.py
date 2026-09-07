"""Unit tests for ``BookerBV2Tool/utils.py``."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import pytest

from BookerBV2Tool import utils


class TestHParams:
    def test_basic_construction(self):
        hp = utils.HParams(a=1, b="x", c=[1, 2])
        assert hp.a == 1
        assert hp.b == "x"
        assert hp.c == [1, 2]

    def test_nested(self):
        hp = utils.HParams(train={"lr": 0.1, "batch": {"size": 8}})
        assert hp.train["lr"] == 0.1

    def test_mapping_protocol(self):
        hp = utils.HParams(a=1, b=2)
        assert hp["a"] == 1
        hp["c"] = 3
        assert hp.c == 3
        assert "a" in hp
        assert "z" not in hp
        assert len(hp) == 3

    def test_items_values_keys(self):
        hp = utils.HParams(a=1, b="y")
        assert set(hp.keys()) == {"a", "b"}
        assert set(hp.values()) == {1, "y"}
        assert dict(hp.items()) == {"a": 1, "b": "y"}

    def test_repr(self):
        hp = utils.HParams(a=1)
        assert "'a'" in repr(hp)


class TestFilePaths:
    def test_load_filepaths_and_text(self, tmp_path):
        f = tmp_path / "list.txt"
        f.write_text("a.wav|A|你好|n i2\nb.wav|B|hello|h\n", encoding="utf-8")
        rows = utils.load_filepaths_and_text(str(f))
        assert rows == [["a.wav", "A", "你好", "n i2"], ["b.wav", "B", "hello", "h"]]

    def test_custom_separator(self, tmp_path):
        f = tmp_path / "list.txt"
        f.write_text("a\tb", encoding="utf-8")
        assert utils.load_filepaths_and_text(str(f), split="\t") == [["a", "b"]]

    def test_blank_lines_are_kept_as_stripped_rows(self, tmp_path):
        f = tmp_path / "list.txt"
        f.write_text("", encoding="utf-8")
        assert utils.load_filepaths_and_text(str(f)) == []


class TestLatestCheckpoint:
    def test_numeric_sort(self, tmp_path):
        for name in ("G_1.pth", "G_10.pth", "G_2.pth", "G_100.pth"):
            (tmp_path / name).touch()
        assert utils.latest_checkpoint_path(str(tmp_path)) == str(tmp_path / "G_100.pth")

    def test_custom_regex(self, tmp_path):
        (tmp_path / "D_5.pth").touch()
        (tmp_path / "D_55.pth").touch()
        assert utils.latest_checkpoint_path(str(tmp_path), regex="D_*.pth") == str(
            tmp_path / "D_55.pth"
        )


class TestCleanCheckpoints:
    @pytest.fixture
    def ckpt_dir(self, tmp_path):
        names = []
        for prefix in ("G", "D"):
            for i in range(1, 7):
                names.append(tmp_path / f"{prefix}_{i}.pth")
            (tmp_path / f"{prefix}_0.pth").touch()
        for p in names:
            p.touch()
        return tmp_path

    def test_keeps_last_two_and_zero(self, ckpt_dir):
        utils.clean_checkpoints(str(ckpt_dir), n_ckpts_to_keep=2, sort_by_time=False)
        remaining = {p.name for p in ckpt_dir.iterdir()}
        assert remaining == {
            "G_5.pth", "G_6.pth", "G_0.pth",
            "D_5.pth", "D_6.pth", "D_0.pth",
        }

    def test_keeps_all_when_under_limit(self, ckpt_dir):
        utils.clean_checkpoints(str(ckpt_dir), n_ckpts_to_keep=10, sort_by_time=False)
        assert len([p for p in ckpt_dir.iterdir()]) == 14

    def test_wd_files_crash_name_key(self, tmp_path):
        # Characterization: clean_checkpoints is meant to prune WD_*.pth too, but
        # its name_key regex `._(\d+)\.pth` cannot match the two-char "WD"
        # prefix, so it raises AttributeError the moment a WD file must be sorted.
        for name in ("G_1.pth", "WD_1.pth"):
            (tmp_path / name).touch()
        with pytest.raises(AttributeError):
            utils.clean_checkpoints(str(tmp_path), n_ckpts_to_keep=0, sort_by_time=False)


class _SimpleModule(torch.nn.Module):
    def __init__(self, a=1.0, jp=2.0, plain=3.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([a], dtype=torch.float32))
        self.ja_bert_proj = torch.nn.Parameter(torch.tensor([jp], dtype=torch.float32))
        self.plain = torch.nn.Parameter(torch.tensor([plain], dtype=torch.float32))


class TestCheckpointIO:
    def test_save_load_roundtrip(self, tmp_path):
        torch.manual_seed(0)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = str(tmp_path / "ckpt.pth")
        utils.save_checkpoint(model, optimizer, learning_rate=1e-3, iteration=42, checkpoint_path=path)

        model2 = torch.nn.Linear(2, 2)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1.0)
        loaded, opt, lr, iteration = utils.load_checkpoint(path, model2, optimizer2)
        assert iteration == 42
        assert lr == 1e-3
        torch.testing.assert_close(model2.weight, model.weight)
        assert opt is optimizer2
        assert optimizer2.param_groups[0]["lr"] == 1e-3

    def test_skip_optimizer_leaves_it_alone(self, tmp_path):
        torch.manual_seed(1)
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        path = str(tmp_path / "ckpt.pth")
        utils.save_checkpoint(model, opt, 0.5, 7, path)

        model2 = torch.nn.Linear(2, 2)
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.9)
        utils.load_checkpoint(path, model2, opt2, skip_optimizer=True)
        assert opt2.param_groups[0]["lr"] == 0.9

    def test_missing_keys_fallback(self, tmp_path, caplog):
        model = _SimpleModule(a=1.0, jp=2.0, plain=3.0)
        # checkpoint has only 'a'
        saved = {
            "model": {"a": torch.tensor([9.0])},
            "iteration": 0,
            "learning_rate": 0.0,
            "optimizer": None,
        }
        path = str(tmp_path / "partial.pth")
        torch.save(saved, path)
        loaded, _, _, _ = utils.load_checkpoint(path, model, None, skip_optimizer=True)
        assert loaded.a.item() == 9.0
        # ja_bert_proj zeroed for backward compatibility
        assert loaded.ja_bert_proj.item() == 0.0
        # other missing keys keep their current value, logged as error
        assert loaded.plain.item() == 3.0
        assert any(r.levelname == "ERROR" and "plain" in r.getMessage() for r in caplog.records)

    def test_shape_mismatch_falls_back(self, tmp_path):
        model = _SimpleModule(a=1.0)
        saved = {
            "model": {"a": torch.zeros(5)},  # wrong shape
            "iteration": 0,
            "learning_rate": 0.0,
            "optimizer": None,
        }
        path = str(tmp_path / "shape.pth")
        torch.save(saved, path)
        loaded, _, _, _ = utils.load_checkpoint(path, model, None, skip_optimizer=True)
        assert loaded.a.item() == 1.0  # keeps original value

    def test_requires_existing_file(self, tmp_path):
        with pytest.raises(AssertionError):
            utils.load_checkpoint(str(tmp_path / "nope.pth"), _SimpleModule(), None, skip_optimizer=True)


class TestSummarize:
    def test_routes_to_writer(self):
        writer = MagicMock()
        utils.summarize(
            writer,
            10,
            scalars={"loss": 0.5},
            histograms={"w": MagicMock()},
            images={"i": MagicMock()},
            audios={"a": MagicMock()},
            audio_sampling_rate=22050,
        )
        writer.add_scalar.assert_called_once_with("loss", 0.5, 10)
        writer.add_histogram.assert_called_once()
        writer.add_image.assert_called_once()
        writer.add_audio.assert_called_once_with("a", writer.add_audio.call_args.args[1], 10, 22050)

    def test_empty_is_ok(self):
        writer = MagicMock()
        utils.summarize(writer, 1)
        writer.add_scalar.assert_not_called()
        writer.add_histogram.assert_not_called()


class TestMixModel:
    class Net(torch.nn.Module):
        def __init__(self, voice, enc):
            super().__init__()
            self.voice = torch.nn.Parameter(torch.tensor([voice], dtype=torch.float32))
            self.enc_p = torch.nn.Parameter(torch.tensor([enc], dtype=torch.float32))

    def test_blend_and_copy(self, tmp_path):
        n1 = self.Net(1.0, 10.0)
        n2 = self.Net(2.0, 20.0)
        n2.extra = torch.nn.Parameter(torch.tensor([99.0], dtype=torch.float32))
        out = str(tmp_path / "mix.pth")
        utils.mix_model(n1, n2, out, voice_ratio=(0.4, 0.6), tone_ratio=(0.3, 0.7))
        ckpt = torch.load(out, map_location="cpu")
        state = ckpt["model"]
        assert state["voice"].item() == pytest.approx(1.0 * 0.4 + 2.0 * 0.6)
        assert state["enc_p"].item() == pytest.approx(10.0 * 0.3 + 20.0 * 0.7)
        assert state["extra"].item() == 99.0
        assert ckpt["iteration"] == 0


class TestGetSteps:
    def test_from_path(self):
        assert utils.get_steps("logs/44k/G_100.pth") == "100"
        assert utils.get_steps("G_000001.pth") == "000001"

    def test_no_digits(self):
        assert utils.get_steps("G.pth") is None


class TestGetHParams:
    def test_from_dir(self, tmp_path):
        import json

        cfg = {"data": {"n_speakers": 2, "spk2id": {"A": 0}}, "train": {"lr": 1e-4}}
        (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        hp = utils.get_hparams_from_dir(str(tmp_path))
        assert hp.data.n_speakers == 2
        # nested dicts become HParams instances (not plain dicts)
        assert hp.data.spk2id.A == 0
        assert hp["train"]["lr"] == 1e-4
        assert hp.model_dir == str(tmp_path)


class TestCheckGitHash:
    def test_non_git_repo_warns(self, tmp_path, monkeypatch, caplog):
        def fake_exists(p):
            return not (str(p).split("\\")[-1] == ".git" or str(p).endswith(".git"))

        monkeypatch.setattr(utils.os.path, "exists", fake_exists)
        with caplog.at_level("WARNING", logger="BookerBV2Tool.utils"):
            utils.check_git_hash(str(tmp_path))
        assert any("git" in r.getMessage() for r in caplog.records)