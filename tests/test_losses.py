"""Unit tests for ``BookerBV2Tool/losses.py``."""

from types import SimpleNamespace

import pytest
import torch

from BookerBV2Tool import losses


def _rand(shape, seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape)


def _fmap(shapes):
    """Build a fake (layers, batched) feature-map list like a discriminator."""
    return [
        [_rand(s, seed=seed) for seed, s in enumerate(shapes)],
        [_rand(s, seed=seed + 10) for seed, s in enumerate(shapes)],
    ]


class TestFeatureLoss:
    def test_zero_for_equal_maps(self):
        torch.manual_seed(0)
        f = _rand((2, 4, 8), 1)
        comes = _fmap([(2, 4, 8)])
        # identical maps -> zero loss
        out = losses.feature_loss(comes, comes)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_and_formula(self):
        torch.manual_seed(0)
        fmap_r = _fmap([(2, 4, 8)])
        fmap_g = _fmap([(2, 4, 8)])
        out = losses.feature_loss(fmap_r, fmap_g)
        total = sum(
            torch.mean(torch.abs(rl.float().detach() - gl.float()))
            for dr, dg in zip(fmap_r, fmap_g)
            for rl, gl in zip(dr, dg)
        )
        assert out.item() == pytest.approx((total * 2).item())


class TestDiscriminatorLoss:
    def test_structure(self):
        torch.manual_seed(1)
        dr = [_rand((2, 1)) for _ in range(3)]
        dg = [_rand((2, 1)) for _ in range(3)]
        loss, r_losses, g_losses = losses.discriminator_loss(dr, dg)
        assert len(r_losses) == len(g_losses) == 3
        # real outputs pushed toward 1, generated toward 0
        expected = sum(torch.mean((1 - d) ** 2) + torch.mean(g ** 2) for d, g in zip(dr, dg))
        assert loss.item() == pytest.approx(expected.item())

    def test_ideal_scores(self):
        dr = [torch.ones(4, 1)]
        dg = [torch.zeros(4, 1)]
        loss, r_losses, g_losses = losses.discriminator_loss(dr, dg)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestGeneratorLoss:
    def test_formula(self):
        torch.manual_seed(2)
        outs = [_rand((4, 1)) for _ in range(2)]
        loss, gen_losses = losses.generator_loss(outs)
        assert len(gen_losses) == 2
        expected = sum(torch.mean((1 - d) ** 2) for d in outs)
        assert loss.item() == pytest.approx(expected.item())

    def test_ideal_generator(self):
        # Generated outputs of 1 -> zero loss
        loss, gen_losses = losses.generator_loss([torch.ones(6, 1)])
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestKLLoss:
    def test_free_unit_gaussians_return_masked_mean_of_minus_half(self):
        # Unlike commons.kl_divergence, losses.kl_loss drops the exp(2*logs_p)
        # normalization term, so identical N(0,1) inputs yield -0.5 per cell.
        torch.manual_seed(3)
        z = _rand((2, 3, 5), 4)
        mask = torch.ones(2, 3, 5)
        logp = torch.zeros(2, 3, 5)
        logq = torch.zeros(2, 3, 5)
        out = losses.kl_loss(z, logq, z, logp, mask)
        assert out.item() == pytest.approx(-0.5)

    def test_value_matches_analytic(self):
        torch.manual_seed(5)
        z_p = _rand((2, 3, 5), 6)
        logs_p = torch.zeros(2, 3, 5)
        logs_q = torch.zeros(2, 3, 5)
        m_p = _rand((2, 3, 5), 7)
        mask = torch.ones(2, 3, 5)
        out = losses.kl_loss(z_p, logs_q, m_p, logs_p, mask)
        kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
        expected = kl.sum() / mask.sum()
        assert out.item() == pytest.approx(expected.item())


class _FakeWavLM:
    """Stand-in for ``transformers.AutoModelForMaskedLM`` (no download)."""

    def __init__(self, n_hidden=3):
        self.n_hidden = n_hidden
        self._eval_called = 0
        self.training = True

    def eval(self):
        self._eval_called += 1
        self.training = False
        return self

    def parameters(self):
        return iter(())

    def requires_grad_(self, *a, **k):
        return self

    def __call__(self, input_values=None, output_hidden_states=False):
        return SimpleNamespace(hidden_states=[input_values] * self.n_hidden)


class _FakeResample:
    """Identity stand-in for ``torchaudio.transforms.Resample``."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x


class _FakeWD:
    """Fake WavLM discriminator head."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return torch.zeros(x.shape[0], 1)


@pytest.fixture
def wavlm_loss(monkeypatch):
    fake_auto = type("AutoModel", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeWavLM())})
    monkeypatch.setattr(losses, "AutoModel", fake_auto)
    monkeypatch.setattr(
        losses.torchaudio.transforms, "Resample", _FakeResample
    )
    wd = _FakeWD()
    return losses.WavLMLoss("fake/model", wd, 16000), wd


class TestWavLMLoss:
    def test_init_freezes_params(self, monkeypatch):
        fake_auto = type(
            "AutoModel", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeWavLM())}
        )
        monkeypatch.setattr(losses, "AutoModel", fake_auto)
        monkeypatch.setattr(losses.torchaudio.transforms, "Resample", _FakeResample)
        m = losses.WavLMLoss("fake", _FakeWD(), 16000, slm_sr=16000)
        assert m.wavlm._eval_called == 1
        assert not m.wavlm.training

    def test_forward_equals_per_layer_l1(self, wavlm_loss):
        lm, _ = wavlm_loss
        torch.manual_seed(10)
        wav = _rand((1, 1600), 11)
        y_rec = _rand((1, 1600), 12)
        out = lm(wav.clone(), y_rec.clone())
        # FakeWavLM broadcasts its input as each of the 3 hidden states, so the
        # loss is 3 * mean(|wav - y_rec|).
        per_layer = torch.mean(torch.abs(wav - y_rec))
        assert out.item() == pytest.approx((3 * per_layer).item())

    def test_generator_trains_wd(self, wavlm_loss):
        lm, wd = wavlm_loss
        torch.manual_seed(13)
        y_rec = _rand((1, 1600), 14)
        loss_gen = lm.generator(y_rec.clone())
        assert wd.calls == 1
        assert loss_gen.item() == pytest.approx(1.0)  # wd returns zeros -> (1-0)^2

    def test_discriminator(self, wavlm_loss):
        lm, wd = wavlm_loss
        torch.manual_seed(15)
        wav = _rand((1, 1600), 16)
        y_rec = _rand((1, 1600), 17)
        out = lm.discriminator(wav.clone(), y_rec.clone())
        # real pushed to 1, fake to 0, wd returns 0 -> r_loss=1, g_loss=0
        assert out.item() == pytest.approx(1.0)

    def test_discriminator_forward(self, wavlm_loss):
        lm, wd = wavlm_loss
        torch.manual_seed(18)
        wav = _rand((1, 1600), 19)
        out = lm.discriminator_forward(wav.clone())
        assert wd.calls == 1
        assert out.shape == (1, 1)