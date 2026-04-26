import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main(checkpoint_path: str, save_dir: str | None = None):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    baseline_samples = ckpt["baseline_samples"]
    pinn_samples = ckpt["pinn_samples"]
    hnn_samples = ckpt["hnn_samples"]

    baseline_stats = ckpt["baseline_stats"]
    pinn_stats = ckpt["pinn_stats"]
    hnn_stats = ckpt["hnn_stats"]

    if save_dir is None:
        save_dir = "plots_compare_pinn_hnn"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1. scatter comparison
    # -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(baseline_samples[:, 0], baseline_samples[:, 1], s=4, alpha=0.25)
    axes[0].set_title("Baseline HMC")
    axes[0].set_xlabel("q1")
    axes[0].set_ylabel("q2")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(pinn_samples[:, 0], pinn_samples[:, 1], s=4, alpha=0.25)
    axes[1].set_title("PINN-HMC")
    axes[1].set_xlabel("q1")
    axes[1].set_ylabel("q2")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(hnn_samples[:, 0], hnn_samples[:, 1], s=4, alpha=0.25)
    axes[2].set_title("HNNMC-style")
    axes[2].set_xlabel("q1")
    axes[2].set_ylabel("q2")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "scatter_compare.png", dpi=200)
    plt.close()

    # -------------------------
    # 2. marginals
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # q1
    axes[0, 0].hist(baseline_samples[:, 0].numpy(), bins=50, alpha=0.5, density=True, label="Baseline")
    axes[0, 0].hist(pinn_samples[:, 0].numpy(), bins=50, alpha=0.5, density=True, label="PINN")
    axes[0, 0].hist(hnn_samples[:, 0].numpy(), bins=50, alpha=0.5, density=True, label="HNN")
    axes[0, 0].set_title("q1 marginal")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # q2
    axes[0, 1].hist(baseline_samples[:, 1].numpy(), bins=50, alpha=0.5, density=True, label="Baseline")
    axes[0, 1].hist(pinn_samples[:, 1].numpy(), bins=50, alpha=0.5, density=True, label="PINN")
    axes[0, 1].hist(hnn_samples[:, 1].numpy(), bins=50, alpha=0.5, density=True, label="HNN")
    axes[0, 1].set_title("q2 marginal")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # acceptance
    methods = ["Baseline", "PINN", "HNN"]
    accept = [
        baseline_stats["mean_accept_rate"],
        pinn_stats["mean_accept_rate"],
        hnn_stats["mean_accept_rate"],
    ]
    axes[1, 0].bar(methods, accept)
    axes[1, 0].set_title("Mean Accept Rate")
    axes[1, 0].grid(True, alpha=0.3)

    # delta H
    deltaH = [
        baseline_stats["mean_abs_delta_H"],
        pinn_stats["mean_abs_delta_H"],
        hnn_stats["mean_abs_delta_H"],
    ]
    axes[1, 1].bar(methods, deltaH)
    axes[1, 1].set_title("Mean |ΔH|")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "metrics_compare.png", dpi=200)
    plt.close()

    print(f"Saved plots to {save_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    main(args.checkpoint, args.save_dir)