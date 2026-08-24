from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import autograd, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .config import load_yaml
from .dataset import dataset_from_config
from .models import build_models
from .utils import seed_everything, select_device, worker_seed


def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    batch = real.shape[0]
    alpha = torch.rand(batch, 1, 1, 1, device=real.device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score = discriminator(interpolated)
    gradients = autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


@torch.no_grad()
def save_samples(
    generators: tuple[nn.Module, nn.Module],
    batch: dict,
    output: Path,
    step: int,
    device: torch.device,
) -> None:
    generator_ab, generator_ba = generators
    was_training = generator_ab.training, generator_ba.training
    generator_ab.eval()
    generator_ba.eval()
    real_a = batch["A"].to(device)[:4]
    real_b = batch["B"].to(device)[:4]
    fake_b = generator_ab(real_a)
    fake_a = generator_ba(real_b)
    grid = torch.cat((real_a, fake_b, real_b, fake_a), dim=0)
    output.mkdir(parents=True, exist_ok=True)
    save_image(grid, output / f"step_{step:08d}.png", nrow=4, normalize=True, value_range=(-1, 1))
    generator_ab.train(was_training[0])
    generator_ba.train(was_training[1])


def save_checkpoint(
    path: Path,
    epoch: int,
    step: int,
    models: tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    optimizers: tuple[Adam, Adam, Adam],
    train_config: dict[str, Any],
) -> None:
    generator_ab, generator_ba, discriminator_a, discriminator_b = models
    optimizer_g, optimizer_da, optimizer_db = optimizers
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "epoch": epoch,
            "step": step,
            "generator_ab": generator_ab.state_dict(),
            "generator_ba": generator_ba.state_dict(),
            "discriminator_a": discriminator_a.state_dict(),
            "discriminator_b": discriminator_b.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_da": optimizer_da.state_dict(),
            "optimizer_db": optimizer_db.state_dict(),
            "train_config": {k: v for k, v in train_config.items() if not k.startswith("_")},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    models: tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    optimizers: tuple[Adam, Adam, Adam],
    device: torch.device,
) -> tuple[int, int]:
    checkpoint_data = torch.load(path, map_location=device, weights_only=False)
    keys = ("generator_ab", "generator_ba", "discriminator_a", "discriminator_b")
    for model, key in zip(models, keys, strict=True):
        model.load_state_dict(checkpoint_data[key])
    optimizer_keys = ("optimizer_g", "optimizer_da", "optimizer_db")
    for optimizer, key in zip(optimizers, optimizer_keys, strict=True):
        optimizer.load_state_dict(checkpoint_data[key])
    return int(checkpoint_data.get("epoch", -1)) + 1, int(checkpoint_data.get("step", 0))


def train(data_config: dict[str, Any], train_config: dict[str, Any]) -> None:
    seed = int(train_config.get("seed", 42))
    seed_everything(seed)
    device = select_device(str(train_config.get("device", "cuda")))
    output_dir = Path(train_config.get("output_dir", "runs/paper_model"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "data": {k: v for k, v in data_config.items() if not k.startswith("_")},
                "train": {k: v for k, v in train_config.items() if not k.startswith("_")},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_dataset = dataset_from_config(data_config, "train", augment=True)
    val_dataset = dataset_from_config(data_config, "val", augment=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(train_config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=worker_seed,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    val_batch = next(iter(val_loader))

    models = tuple(model.to(device) for model in build_models(train_config))
    generator_ab, generator_ba, discriminator_a, discriminator_b = models
    learning_rate = float(train_config.get("learning_rate", 2e-4))
    betas = (float(train_config.get("beta1", 0.5)), float(train_config.get("beta2", 0.999)))
    optimizer_g = Adam(
        list(generator_ab.parameters()) + list(generator_ba.parameters()),
        lr=learning_rate,
        betas=betas,
    )
    optimizer_da = Adam(discriminator_a.parameters(), lr=learning_rate, betas=betas)
    optimizer_db = Adam(discriminator_b.parameters(), lr=learning_rate, betas=betas)
    optimizers = optimizer_g, optimizer_da, optimizer_db

    start_epoch, global_step = 0, 0
    if train_config.get("resume"):
        start_epoch, global_step = load_checkpoint(
            train_config["resume"], models, optimizers, device
        )

    cycle_loss = nn.L1Loss()
    lambda_cycle = float(train_config.get("lambda_cycle", 10.0))
    lambda_gp = float(train_config.get("lambda_gp", 10.0))
    n_critic = int(train_config.get("n_critic", 5))
    sample_interval = int(train_config.get("sample_interval", 500))
    checkpoint_interval = int(train_config.get("checkpoint_interval", 10))
    metrics_path = output_dir / "metrics.jsonl"

    for epoch in range(start_epoch, int(train_config.get("epochs", 300))):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        for batch_index, batch in enumerate(progress):
            real_a = batch["A"].to(device, non_blocking=True)
            real_b = batch["B"].to(device, non_blocking=True)

            optimizer_da.zero_grad(set_to_none=True)
            optimizer_db.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_b_detached = generator_ab(real_a)
                fake_a_detached = generator_ba(real_b)
            loss_da = (
                -discriminator_a(real_a).mean()
                + discriminator_a(fake_a_detached).mean()
                + lambda_gp * gradient_penalty(discriminator_a, real_a, fake_a_detached)
            )
            loss_db = (
                -discriminator_b(real_b).mean()
                + discriminator_b(fake_b_detached).mean()
                + lambda_gp * gradient_penalty(discriminator_b, real_b, fake_b_detached)
            )
            (loss_da + loss_db).backward()
            optimizer_da.step()
            optimizer_db.step()

            loss_g = torch.zeros((), device=device)
            loss_adv = torch.zeros((), device=device)
            loss_cycle = torch.zeros((), device=device)
            if batch_index % n_critic == 0:
                optimizer_g.zero_grad(set_to_none=True)
                fake_b = generator_ab(real_a)
                fake_a = generator_ba(real_b)
                recovered_a = generator_ba(fake_b)
                recovered_b = generator_ab(fake_a)
                loss_adv = -discriminator_b(fake_b).mean() - discriminator_a(fake_a).mean()
                loss_cycle = cycle_loss(recovered_a, real_a) + cycle_loss(recovered_b, real_b)
                loss_g = loss_adv + lambda_cycle * loss_cycle
                loss_g.backward()
                optimizer_g.step()

            global_step += 1
            values = {
                "time": time.time(),
                "epoch": epoch + 1,
                "step": global_step,
                "loss_da": float(loss_da.detach()),
                "loss_db": float(loss_db.detach()),
                "loss_g": float(loss_g.detach()),
                "loss_adv": float(loss_adv.detach()),
                "loss_cycle": float(loss_cycle.detach()),
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(values) + "\n")
            progress.set_postfix(g=f"{values['loss_g']:.3f}", d=f"{values['loss_da'] + values['loss_db']:.3f}")
            if sample_interval > 0 and global_step % sample_interval == 0:
                save_samples(
                    (generator_ab, generator_ba),
                    val_batch,
                    output_dir / "samples",
                    global_step,
                    device,
                )

        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
                epoch,
                global_step,
                models,
                optimizers,
                train_config,
            )
        save_checkpoint(
            output_dir / "checkpoints" / "latest.pt",
            epoch,
            global_step,
            models,
            optimizers,
            train_config,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train bidirectional heterogeneous SAR translation")
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(load_yaml(args.data_config), load_yaml(args.train_config))


if __name__ == "__main__":
    main()
