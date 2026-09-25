"""Unit tests for ``infer.py`` — text-to-speech inference (version 2.3)."""


def test_infer_module_imports():
    import BookerBV2Tool.infer as inf

    assert inf.latest_version == "2.3"
    assert hasattr(inf, "get_net_g")
    assert hasattr(inf, "get_text")
    assert hasattr(inf, "infer")
    assert hasattr(inf, "infer_handle")


def test_get_text_shapes_and_blank(monkeypatch):
    from types import SimpleNamespace

    import torch

    import BookerBV2Tool.infer as inf
    from BookerBV2Tool import commons

    # clean_text returns 2 phones that become 2*2+1 = 5 after intersperse
    fake_clean = lambda model_name, text, lang: ("NORM", ["a", "i"], [1, 2], [1, 1])
    monkeypatch.setattr(inf, "clean_text", fake_clean)
    monkeypatch.setattr(
        inf, "get_bert_feature", lambda model_name, text, word2ph, device, style_text=None, style_weight=0.7: torch.zeros(1024, 5)
    )

    hps = SimpleNamespace(data=SimpleNamespace(add_blank=True))
    args = SimpleNamespace(chinese_bert="cb", enlish_bert="eb", japanese_bert="jb")
    bert, ja_bert, en_bert, phone, tone, language = inf.get_text(
        "some text", "ZH", hps, torch.device("cpu"), args
    )
    assert phone.shape == (5,)
    assert tone.shape == (5,)
    assert language.shape == (5,)
    assert bert.shape == (1024, 5)
    assert ja_bert.shape == (1024, 5)
    assert en_bert.shape == (1024, 5)


def test_get_net_g_rejects_old_version(monkeypatch):
    import torch

    from BookerBV2Tool import infer as inf

    try:
        inf.get_net_g("nonexistent.pth", "1.1", torch.device("cpu"), object())
        raise AssertionError("should have raised ValueError for old version")
    except ValueError:
        pass