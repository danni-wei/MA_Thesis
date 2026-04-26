import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_history(checkpoint_path: str, save_dir: str | None = None):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    history = ckpt["history"]
    method = ckpt.get("method", Path(checkpoint_path).stem)

    epochs = [h["epoch"] for h in history]

    train_total = [h["train_total_loss"] for h in history]
    val_total = [h["val_total_loss"] for h in history]

    train_sup = [h["train_sup"] for h in history]
    train_phys = [h["train_phys"] for h in history]
    train_ic = [h["train_ic"] for h in history]

    val_sup = [h["val_sup"] for h in history]
    val_phys = [h["val_phys"] for h in history]
    val_ic = [h["val_ic"] for h in history]

    train_lambda_sup = [h["train_lambda_sup"] for h in history]
    train_lambda_phys = [h["train_lambda_phys"] for h in history]
    train_lambda_ic = [h["train_lambda_ic"] for h in history]

    if save_dir is None:
        save_dir = f"plots_{method}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Figure 1: total loss
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_total, label="train total")
    plt.plot(epochs, val_total, label="val total")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Total Loss Convergence ({method})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{method}_total_loss.png", dpi=200)
    plt.close()

    # --------------------------------------------------
    # Figure 2: raw train losses
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_sup, label="train sup")
    plt.plot(epochs, train_phys, label="train phys")
    plt.plot(epochs, train_ic, label="train ic")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train Raw Losses ({method})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{method}_train_raw_losses.png", dpi=200)
    plt.close()

    # --------------------------------------------------
    # Figure 3: raw val losses
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, val_sup, label="val sup")
    plt.plot(epochs, val_phys, label="val phys")
    plt.plot(epochs, val_ic, label="val ic")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Validation Raw Losses ({method})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{method}_val_raw_losses.png", dpi=200)
    plt.close()

    # --------------------------------------------------
    # Figure 4: lambda evolution
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_lambda_sup, label="lambda_sup")
    plt.plot(epochs, train_lambda_phys, label="lambda_phys")
    plt.plot(epochs, train_lambda_ic, label="lambda_ic")
    plt.xlabel("Epoch")
    plt.ylabel("Lambda")
    plt.title(f"Lambda Evolution ({method})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{method}_lambdas.png", dpi=200)
    plt.close()

    print(f"Saved plots to: {save_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_*.pt")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()

    plot_history(args.checkpoint, args.save_dir)