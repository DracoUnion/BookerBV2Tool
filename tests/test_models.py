"""Unit tests for the bundled Bert-VITS2 model library.

These modules come from the upstream ``Bert-VITS2`` repo (modules/attentions/
transforms/monotonic_align) and are imported via relative paths inside the
``BookerBV2Tool`` package.  Prior to bundling, ``models.py`` could not even be
imported because those files were absent.
"""


def test_model_chain_imports():
    # The full training chain must import cleanly.
    import BookerBV2Tool.train  # noqa: F401
    from BookerBV2Tool.models import (  # noqa: F401
        SynthesizerTrn,
        MultiPeriodDiscriminator,
        DurationDiscriminator,
        WavLMDiscriminator,
        Generator,
    )


def test_attentions_transforms_modules_import():
    import BookerBV2Tool.attentions as a
    import BookerBV2Tool.transforms as tr
    import BookerBV2Tool.monotonic_align as ma

    assert hasattr(a, "Encoder")
    assert hasattr(tr, "piecewise_rational_quadratic_transform")
    assert callable(ma.maximum_path)


def test_synthesizer_trn_constructs():
    import torch

    from BookerBV2Tool.models import SynthesizerTrn

    torch.manual_seed(0)
    model = SynthesizerTrn(
        32,  # n_vocab
        129,  # spec_channels
        8,  # segment_size
        inter_channels=32,
        hidden_channels=32,
        filter_channels=128,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2],
        upsample_initial_channel=64,
        upsample_kernel_sizes=[16, 16, 4],
        n_layers_trans_flow=4,
        n_speakers=2,
        use_spk_conditioned_encoder=False,
        use_sdp=True,
        use_noise_scaled_mas=True,
    )
    model.eval()
    assert len(list(model.parameters())) > 0


def test_synthesizer_trn_conditioned_encoder():
    # With use_spk_conditioned_encoder=True and gin_channels>0, enc_gin_channels
    # must still be initialized (regression for the missing-attribute bug).
    import torch

    from BookerBV2Tool.models import SynthesizerTrn

    torch.manual_seed(0)
    model = SynthesizerTrn(
        32,
        129,
        8,
        inter_channels=32,
        hidden_channels=32,
        filter_channels=128,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2],
        upsample_initial_channel=64,
        upsample_kernel_sizes=[16, 16, 4],
        n_layers_trans_flow=4,
        n_speakers=2,
        use_spk_conditioned_encoder=True,
        use_sdp=True,
    )
    assert isinstance(model.enc_gin_channels, int)
    assert len(list(model.parameters())) > 0