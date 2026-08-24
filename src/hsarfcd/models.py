from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm2d) and module.affine:
        if module.weight is not None:
            nn.init.normal_(module.weight, 1.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.extend(
            [
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, 3),
                nn.InstanceNorm2d(channels, affine=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MultiDilatedResidualBlock(nn.Module):
    """Parallel 3x3 atrous convolutions with dilation rates 1, 2 and 4."""

    def __init__(self, channels: int, dropout: bool = False) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=rate, dilation=rate),
                    nn.InstanceNorm2d(channels, affine=True),
                    nn.ReLU(inplace=True),
                )
                for rate in (1, 2, 4)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        return x + self.dropout(self.fuse(features))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(channels * 3, channels, 1)
        self.mask = nn.Sequential(nn.Conv2d(channels, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, branch1: torch.Tensor, branch2: torch.Tensor, branch3: torch.Tensor) -> torch.Tensor:
        fused = self.reduce(torch.cat([branch1, branch2, branch3], dim=1))
        attention = self.mask(fused)
        average = (branch1 + branch2 + branch3) / 3.0
        return average * (1.0 + attention)


class MultiScaleGenerator(nn.Module):
    """Paper generator: parallel encoding, atrous fusion, three decoders and spatial attention."""

    def __init__(
        self,
        channels: int = 3,
        base_channels: int = 64,
        residual_blocks: int = 9,
        dropout: bool = False,
        checkpoint_blocks: bool = False,
    ) -> None:
        super().__init__()
        features = base_channels * 4
        self.channels = channels
        self.checkpoint_blocks = checkpoint_blocks
        self.shallow = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, base_channels, 7),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, features, 3, padding=1),
            nn.InstanceNorm2d(features, affine=True),
            nn.ReLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        blocks: list[nn.Module] = []
        for index in range(residual_blocks):
            block_type: Callable[..., nn.Module]
            block_type = MultiDilatedResidualBlock if index % 3 == 0 else ResidualBlock
            blocks.append(block_type(features, dropout=dropout))
        self.residual = nn.Sequential(*blocks)

        self.branch1_up = nn.Sequential(UpsampleBlock(features, features), UpsampleBlock(features, features))
        self.branch1_out = self._head(features, base_channels, channels, depth=3)

        self.branch2_up = UpsampleBlock(features, features)
        self.branch2_out = self._head(features, base_channels * 2, channels, depth=2)

        self.branch3_up = nn.Sequential(UpsampleBlock(features, features), UpsampleBlock(features, features))
        self.branch3_out = self._head(features, base_channels, channels, depth=1)

        self.attention = SpatialAttention(channels)
        self.output = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.Tanh(),
        )
        self.apply(initialize_weights)

    @staticmethod
    def _head(in_channels: int, hidden: int, out_channels: int, depth: int) -> nn.Sequential:
        if depth == 1:
            return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_channels, out_channels, 3))
        layers: list[nn.Module] = []
        current = in_channels
        for _ in range(depth - 1):
            kernel = 7 if depth == 3 else 3
            pad = kernel // 2
            layers.extend(
                [
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(current, hidden, kernel),
                    nn.InstanceNorm2d(hidden, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            current = hidden
        kernel = 7 if depth == 3 else 3
        layers.extend([nn.ReflectionPad2d(kernel // 2), nn.Conv2d(current, out_channels, kernel)])
        return nn.Sequential(*layers)

    def _run(self, module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_blocks and self.training and tensor.requires_grad:
            return checkpoint(module, tensor, use_reentrant=False)
        return module(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("Input height and width must be divisible by 4")
        y1 = self._run(self.shallow, x)
        x2 = self._run(self.down, x)
        y2 = self._run(self.shallow, x2)
        x3 = self._run(self.down, x2)
        y3 = self._run(self.shallow, x3)

        fused = F.max_pool2d(y1, 4) + F.max_pool2d(y2, 2) + y3
        fused = self._run(self.residual, fused)

        branch1 = self.branch1_out(self._run(self.branch1_up, fused) + y1)
        branch2 = self.branch2_out(self._run(self.branch2_up, fused + F.max_pool2d(y2, 2)))
        branch2 = F.interpolate(branch2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        branch3 = self.branch3_out(self._run(self.branch3_up, fused) + y1)
        return self.output(self.attention(branch1, branch2, branch3))


class PatchDiscriminator(nn.Module):
    def __init__(self, channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int, normalize_layer: bool = True):
            layers: list[nn.Module] = [
                spectral_norm(nn.Conv2d(in_channels, out_channels, 4, 2, 1))
            ]
            if normalize_layer:
                layers.append(nn.InstanceNorm2d(out_channels, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(channels, base_channels, False),
            *block(base_channels, base_channels * 2),
            *block(base_channels * 2, base_channels * 4),
            *block(base_channels * 4, base_channels * 8),
            spectral_norm(nn.Conv2d(base_channels * 8, 1, 4, 1, 1)),
        )
        self.apply(initialize_weights)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


def build_models(config: dict) -> tuple[MultiScaleGenerator, MultiScaleGenerator, PatchDiscriminator, PatchDiscriminator]:
    model_config = config.get("model", config)
    channels = int(model_config.get("channels", 3))
    kwargs = {
        "channels": channels,
        "base_channels": int(model_config.get("base_channels", 64)),
        "residual_blocks": int(model_config.get("residual_blocks", 9)),
        "dropout": bool(model_config.get("dropout", False)),
        "checkpoint_blocks": bool(model_config.get("checkpoint_blocks", False)),
    }
    generator_ab = MultiScaleGenerator(**kwargs)
    generator_ba = MultiScaleGenerator(**kwargs)
    discriminator_a = PatchDiscriminator(channels, kwargs["base_channels"])
    discriminator_b = PatchDiscriminator(channels, kwargs["base_channels"])
    return generator_ab, generator_ba, discriminator_a, discriminator_b
