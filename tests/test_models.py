"""Model shapes and the parameter budget."""

import pytest
import torch

from kws.models import DSCNN, DepthwiseSeparableBlock, MFCCLogisticRegression, count_parameters


def test_dscnn_maps_mfcc_to_class_logits():
    model = DSCNN(n_classes=12)
    assert model(torch.randn(8, 1, 49, 10)).shape == (8, 12)


def test_dscnn_fits_the_mcu_parameter_budget():
    """Under 100k parameters, or int8 weights will not fit a small MCU."""
    assert count_parameters(DSCNN()) < 100_000


def test_dscnn_is_in_the_hello_edge_size_range():
    """DS-CNN-S is ~24k parameters; this should be the same order."""
    assert 10_000 < count_parameters(DSCNN()) < 50_000


def test_separable_block_is_far_cheaper_than_a_dense_convolution():
    """The whole reason for depthwise separation."""
    channels = 64
    separable = count_parameters(DepthwiseSeparableBlock(channels))
    dense = channels * channels * 9  # a plain 3x3 over the same channels
    assert separable < dense / 4


def test_width_and_depth_scale_the_parameter_count():
    assert count_parameters(DSCNN(channels=32)) < count_parameters(DSCNN(channels=64))
    assert count_parameters(DSCNN(n_blocks=2)) < count_parameters(DSCNN(n_blocks=4))


def test_baseline_is_linear_and_tiny():
    model = MFCCLogisticRegression(n_mfcc=10, n_classes=12)
    # 2 pooled statistics per coefficient, plus bias.
    assert count_parameters(model) == 2 * 10 * 12 + 12
    assert model(torch.randn(4, 1, 49, 10)).shape == (4, 12)


def test_baseline_ignores_frame_ordering():
    """Mean and standard deviation over time are permutation invariant."""
    model = MFCCLogisticRegression().eval()
    x = torch.randn(2, 1, 49, 10)
    shuffled = x[:, :, torch.randperm(49), :]
    with torch.no_grad():
        assert torch.allclose(model(x), model(shuffled), atol=1e-5)


@pytest.mark.parametrize("batch", [1, 5, 64])
def test_dscnn_handles_any_batch_size(batch):
    assert DSCNN().eval()(torch.randn(batch, 1, 49, 10)).shape == (batch, 12)
