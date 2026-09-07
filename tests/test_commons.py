"""Unit tests for ``BookerBV2Tool/commons.py`` — pure torch helper functions."""

import torch
import pytest

from BookerBV2Tool import commons


def _tensor(*shape, seed=0, dtype=torch.float32, device=None):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=dtype, device=device)


class TestGetPadding:
    def test_kernel_even(self):
        assert commons.get_padding(8, dilation=1) == 3
        assert commons.get_padding(4, dilation=1) == 1

    def test_kernel_odd(self):
        assert commons.get_padding(5, dilation=1) == 2

    def test_dilated(self):
        # (k * d - d) / 2
        assert commons.get_padding(4, dilation=2) == 3


class TestConvertPadShape:
    def test_reverses_and_flattens(self):
        assert commons.convert_pad_shape([[0, 0], [0, 0], [1, 0]]) == [
            1,
            0,
            0,
            0,
            0,
            0,
        ]

    def test_identity(self):
        assert commons.convert_pad_shape([[1, 1], [2, 2]]) == [2, 2, 1, 1]


class TestIntersperse:
    def test_basic(self):
        assert commons.intersperse([1, 2, 3], 0) == [0, 1, 0, 2, 0, 3, 0]

    def test_empty(self):
        assert commons.intersperse([], 0) == [0]

    def test_single(self):
        assert commons.intersperse(["a"], "x") == ["x", "a", "x"]


class TestKLDivergence:
    def test_zero_when_distributions_equal(self):
        m = torch.zeros(2, 3)
        logs = torch.zeros(2, 3)
        kl = commons.kl_divergence(m, logs, m, logs)
        # KL(N(0,1) || N(0,1)) = 0
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)
        assert kl.shape == (2, 3)

    def test_value_matches_analytic_formula(self):
        m_p = torch.tensor([[0.0, 1.0]])
        logs_p = torch.tensor([[0.0, 0.0]])
        m_q = torch.tensor([[1.0, 0.0]])
        logs_q = torch.tensor([[0.0, 1.0]])
        kl = commons.kl_divergence(m_p, logs_p, m_q, logs_q)
        expected = (logs_q - logs_p) - 0.5 + 0.5 * (
            torch.exp(2 * logs_p) + (m_p - m_q) ** 2
        ) * torch.exp(-2 * logs_q)
        assert torch.allclose(kl, expected)


class TestRandGumbel:
    def test_shape_and_dtype(self):
        g = commons.rand_gumbel((4, 5))
        assert g.shape == (4, 5)
        assert g.dtype == torch.float32
        assert torch.isfinite(g).all()

    def test_deterministic_with_seed(self):
        torch.manual_seed(42)
        g1 = commons.rand_gumbel((3, 3))
        torch.manual_seed(42)
        g2 = commons.rand_gumbel((3, 3))
        torch.testing.assert_close(g1, g2)

    def test_matches_inverse_cdf(self):
        # gumbel = -log(-log(u)) with u in [1e-5, 0.99998]; replicate exactly.
        torch.manual_seed(0)
        u = torch.rand(1000) * 0.99998 + 0.00001
        expected = -torch.log(-torch.log(u))
        torch.manual_seed(0)
        got = commons.rand_gumbel((1000,))
        torch.testing.assert_close(got, expected)

    def test_rand_gumbel_like(self):
        x = _tensor(2, 8, device=torch.device("cpu"))
        g = commons.rand_gumbel_like(x)
        assert g.shape == x.shape
        assert g.dtype == x.dtype
        assert g.device == x.device


class TestSliceSegments:
    def test_exact_gather(self):
        x = _tensor(2, 3, 10, seed=1)
        ids = torch.tensor([0, 2])
        out = commons.slice_segments(x, ids, segment_size=4)
        assert out.shape == (2, 3, 4)
        for b in range(2):
            torch.testing.assert_close(out[b], x[b, :, ids[b] : ids[b] + 4])

    def test_rand_slice_segments_bounds_with_lengths(self):
        torch.manual_seed(7)
        x = _tensor(5, 3, 10, seed=2)
        x_lengths = torch.full((5,), 10)
        out, ids = commons.rand_slice_segments(x, x_lengths, segment_size=4)
        assert out.shape == (5, 3, 4)
        assert (ids >= 0).all() and (ids <= 10 - 4).all()

    def test_default_lengths_is_broken_on_torch_2(self):
        # Characterization test: rand_slice_segments() with the default
        # x_lengths=None passes a plain int to torch.clamp(..., min=...),
        # which raises TypeError on torch>=2. Callers must pass a length
        # tensor (as train.py does).
        x = _tensor(2, 3, 10, seed=100)
        with pytest.raises(TypeError):
            commons.rand_slice_segments(x, segment_size=4)

    def test_rand_slice_segments_with_lengths(self):
        torch.manual_seed(7)
        x = _tensor(4, 2, 12, seed=3)
        x_lengths = torch.tensor([12, 8, 6, 12])
        out, ids = commons.rand_slice_segments(x, x_lengths, segment_size=3)
        assert out.shape == (4, 2, 3)
        # ids_str clamped so that id + segment_size <= length
        assert int(ids[1]) + 3 <= 8
        assert int(ids[2]) + 3 <= 6

    def test_rand_slice_segments_zero_length(self):
        # A frame width smaller than the segment size must not crash.
        torch.manual_seed(7)
        x = _tensor(2, 3, 5, seed=4)
        x_lengths = torch.tensor([2, 5])
        out, ids = commons.rand_slice_segments(x, x_lengths, segment_size=4)
        assert out.shape == (2, 3, 4)


class TestTimingSignal:
    def test_get_timing_signal_shape(self):
        sig = commons.get_timing_signal_1d(32, 8)
        assert sig.shape == (1, 8, 32)

    def test_signal_is_sinusoid_based(self):
        sig = commons.get_timing_signal_1d(16, 4)
        assert (sig.abs() <= 1.0 + 1e-6).all()  # sin/cos bounded

    def test_add_timing_signal(self):
        x = _tensor(2, 8, 32, seed=5)
        out = commons.add_timing_signal_1d(x)
        assert out.shape == x.shape
        torch.testing.assert_close(
            out, x + commons.get_timing_signal_1d(32, 8), atol=1e-4, rtol=1e-4
        )

    def test_cat_timing_signal_doubles_channels(self):
        x = _tensor(1, 4, 16, seed=6)
        out = commons.cat_timing_signal_1d(x)
        assert out.shape == (1, 8, 16)


class TestMasks:
    def test_subsequent_mask_lower_triangular(self):
        m = commons.subsequent_mask(4)
        assert m.shape == (1, 1, 4, 4)
        torch.testing.assert_close(m[0, 0], torch.tril(torch.ones(4, 4)))

    def test_sequence_mask(self):
        mask = commons.sequence_mask(torch.tensor([2, 4]))
        assert mask.shape == (2, 4)
        assert mask[0].tolist() == [True, True, False, False]
        assert mask[1].tolist() == [True, True, True, True]

    def test_sequence_mask_with_explicit_max_length(self):
        mask = commons.sequence_mask(torch.tensor([1, 3]), max_length=5)
        assert mask.shape == (2, 5)

    def test_shift_1d(self):
        x = torch.arange(6.0).view(1, 1, 6)
        out = commons.shift_1d(x)
        # shift_1d pads on the left, so out[..., i] == x[..., i - 1]
        assert out.shape == x.shape
        torch.testing.assert_close(out[..., 1:], x[..., :-1])
        assert out[..., 0].item() == 0.0


class TestGeneratePath:
    def test_shape_and_column_masses(self):
        duration = torch.tensor([[[2.0, 1.0, 3.0]]])  # [1, 1, 3]
        mask = torch.ones(1, 1, 6, 3)
        path = commons.generate_path(duration, mask)
        # path is monotonic: sum over time dim equals the durations
        assert path.shape == (1, 1, 6, 3)
        col_sum = path.sum(dim=2)  # [1, 1, 3]
        torch.testing.assert_close(col_sum, duration)


class TestFusedOps:
    def test_fused_add_tanh_sigmoid_multiply(self):
        torch.manual_seed(0)
        a = _tensor(2, 8, 16, seed=8)
        b = _tensor(2, 8, 16, seed=9)
        n_channels = torch.tensor([4], dtype=torch.long)
        out = commons.fused_add_tanh_sigmoid_multiply(a, b, n_channels)
        assert out.shape == (2, 4, 16)
        in_act = a + b
        expected = torch.tanh(in_act[:, :4, :]) * torch.sigmoid(in_act[:, 4:, :])
        torch.testing.assert_close(out, expected)


class TestInitWeights:
    def test_convs_are_reinitialized(self):
        torch.manual_seed(11)
        conv = torch.nn.Conv1d(4, 8, kernel_size=3)
        old = conv.weight.data.clone()
        commons.init_weights(conv, mean=0.0, std=0.01)
        # std 0.01 is much smaller than default conv init (kaiming ~0.33)
        assert conv.weight.data.std().item() < 0.02
        assert not torch.allclose(old, conv.weight.data)

    def test_non_conv_untouched(self):
        lin = torch.nn.Linear(4, 4)
        old = lin.weight.data.clone()
        commons.init_weights(lin)
        torch.testing.assert_close(old, lin.weight.data)


class TestClipGradValue:
    def test_clips_large_gradients(self):
        torch.manual_seed(3)
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = torch.tensor([100.0, -50.0, 0.1, 0.0])
        norm = commons.clip_grad_value_([p], 10.0)
        assert (p.grad.abs() <= 10.0).all()
        assert norm > 0

    def test_filters_none_grads(self):
        p1 = torch.nn.Parameter(torch.zeros(3))
        p2 = torch.nn.Parameter(torch.zeros(3))  # no grad
        p1.grad = torch.full((3,), 0.5)
        norm = commons.clip_grad_value_([p1, p2], 1.0)
        assert torch.allclose(p1.grad, torch.full((3,), 0.5))
        assert norm == pytest.approx((0.5 ** 2 * 3) ** 0.5)

    def test_accepts_single_tensor(self):
        p = torch.nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([-9.0, 9.0])
        commons.clip_grad_value_(p, 5.0)
        assert (p.grad.abs() <= 5.0).all()