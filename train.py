"""
Train a StoryLLM experiment.

Usage:
    uv run python train.py --exp experiments/exp01_post_norm.yaml
    uv run python train.py --exp experiments/exp02_pre_norm.yaml --no-wandb
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from story_llm.config import ModelConfig, TrainConfig
from story_llm.data import get_dataloaders
from story_llm.model import StoryModel


# ---------------------------------------------------------------------------
# Experiment loading
# ---------------------------------------------------------------------------

def load_experiment(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
    use_wandb: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0

    for step, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        loss.backward()
        grad_norm = torch.nn.utils.get_total_norm(
            [p for p in model.parameters() if p.grad is not None]
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        dt = time.perf_counter() - t0
        tokens_per_sec = input_ids.numel() / dt
        global_step = epoch * len(dataloader) + step

        if use_wandb:
            import wandb
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/ppl": torch.exp(loss).item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": float(grad_norm),
                    "perf/tokens_per_sec": tokens_per_sec,
                    "perf/ms_per_step": dt * 1000,
                },
                step=global_step,
            )

        if step % cfg.log_interval == 0:
            print(
                f"  Ep[{epoch + 1}] Step[{step:>5}/{len(dataloader)}] "
                f"loss={loss.item():.4f}  ppl={torch.exp(loss).item():.2f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"gnorm={float(grad_norm):.2f}  "
                f"tok/s={tokens_per_sec:.0f}",
                flush=True,
            )

    avg_loss = total_loss / len(dataloader)
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a StoryLLM experiment")
    parser.add_argument("--exp", required=True, help="Path to experiment YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    cfg_dict = load_experiment(args.exp)

    exp_name: str = cfg_dict["name"]
    description: str = cfg_dict.get("description", "")
    wandb_run_name: str = cfg_dict.get("wandb_run_name", exp_name)
    wandb_project: str = cfg_dict.get("wandb_project", "story-llm")

    train_cfg = TrainConfig(**cfg_dict["train"])
    model_cfg_dict: dict = cfg_dict["model"]  # vocab_size filled after tokenizer

    # Reproducibility
    torch.manual_seed(train_cfg.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Description: {description}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    # Paths
    root = Path(__file__).parent
    data_dir = root / "data"
    tokenizer_dir = root / "tokenizer"
    checkpoint_dir = root / "checkpoints" / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Data
    tokenizer, train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        tokenizer_dir=tokenizer_dir,
        train_size=train_cfg.train_size,
        test_size=train_cfg.test_size,
        max_len=train_cfg.max_len,
        stride=train_cfg.stride,
        batch_size=train_cfg.batch_size,
        seed=train_cfg.seed,
        num_workers=train_cfg.num_workers,
    )

    # Model
    model_cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size(), **model_cfg_dict)
    model = StoryModel(config=model_cfg).to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")
    print(f"Train steps: {train_cfg.num_epochs * len(train_loader):,}\n")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )
    criterion = nn.CrossEntropyLoss()

    total_steps = train_cfg.num_epochs * len(train_loader)
    warmup_steps = int(train_cfg.warmup_ratio * total_steps)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                **model_cfg.__dict__,
                **train_cfg.__dict__,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "n_params": n_params,
                "exp_name": exp_name,
                "description": description,
            },
        )
        wandb.watch(model, log="all", log_freq=10)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(train_cfg.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{train_cfg.num_epochs} ---")
        train_loss, train_ppl = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler,
            device, epoch, train_cfg, use_wandb,
        )
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)

        print(
            f"\n[Epoch {epoch + 1}] "
            f"train_loss={train_loss:.4f}  train_ppl={train_ppl:.2f}  "
            f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}"
        )

        if use_wandb:
            import wandb
            wandb.log(
                {
                    "epoch/train_loss": train_loss,
                    "epoch/train_ppl": train_ppl,
                    "epoch/val_loss": val_loss,
                    "epoch/val_ppl": val_ppl,
                },
                step=(epoch + 1) * len(train_loader),
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = checkpoint_dir / f"best_ep{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_config": model_cfg.__dict__,
                    "train_config": train_cfg.__dict__,
                    "exp_name": exp_name,
                },
                ckpt,
            )
            print(f"  ✓ Checkpoint saved → {ckpt}")

    if use_wandb:
        import wandb
        wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
