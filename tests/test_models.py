import torch

from hsarfcd.models import MultiScaleGenerator, PatchDiscriminator
from hsarfcd.train import gradient_penalty


def test_paper_generator_preserves_shape_and_range():
    model = MultiScaleGenerator(base_channels=8, residual_blocks=3)
    sample = torch.randn(1, 3, 32, 32)
    output = model(sample)
    assert output.shape == sample.shape
    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)


def test_discriminator_and_gradient_penalty_are_finite():
    discriminator = PatchDiscriminator(base_channels=8)
    real = torch.randn(2, 3, 64, 64)
    fake = torch.randn(2, 3, 64, 64)
    score = discriminator(real)
    penalty = gradient_penalty(discriminator, real, fake)
    assert score.ndim == 4
    assert torch.isfinite(penalty)

