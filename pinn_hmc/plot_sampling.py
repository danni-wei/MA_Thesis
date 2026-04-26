import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_sampling(checkpoint_path: str, save_dir: str | None = None) -> None:
    print(">>> Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(">>> Checkpoint loaded")

    method = ckpt.get("method", Path(checkpoint_path).stem)

    if save_dir is None:
        save_dir = f"plots_{method}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> Saving plots to: {save_dir.resolve()}")

    assert "baseline_samples" in ckpt, "No baseline_samples found in checkpoint!"
    assert "pinn_samples" in ckpt, "No pinn_samples found in checkpoint!"

    baseline_samples = ckpt["baseline_samples"]
    pinn_samples = ckpt["pinn_samples"]

    if not isinstance(baseline_samples, torch.Tensor):
        baseline_samples = torch.tensor(baseline_samples)
    if not isinstance(pinn_samples, torch.Tensor):
        pinn_samples = torch.tensor(pinn_samples)

    baseline_samples = baseline_samples.detach().cpu()
    pinn_samples = pinn_samples.detach().cpu()

    print("Baseline samples shape:", tuple(baseline_samples.shape))
    print("PINN samples shape    :", tuple(pinn_samples.shape))

    # --------------------------------------------------
    # 1. Scatter plot
    # --------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.scatter(
        baseline_samples[:, 0],
        baseline_samples[:, 1],
        s=5,
        alpha=0.25,
        label="Baseline HMC",
    )
    plt.scatter(
        pinn_samples[:, 0],
        pinn_samples[:, 1],
        s=5,
        alpha=0.25,
        label="PINN-HMC",
    )
    plt.xlabel("q1")
    plt.ylabel("q2")
    plt.title(f"Sample Scatter ({method})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = save_dir / f"{method}_scatter.png"
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f">>> Saved scatter plot: {scatter_path.resolve()}")

    # --------------------------------------------------
    # 2. Marginal histograms
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        baseline_samples[:, 0].numpy(),
        bins=50,
        alpha=0.5,
        label="Baseline HMC",
        density=True,
    )
    axes[0].hist(
        pinn_samples[:, 0].numpy(),
        bins=50,
        alpha=0.5,
        label="PINN-HMC",
        density=True,
    )
    axes[0].set_title(f"Marginal of q1 ({method})")
    axes[0].set_xlabel("q1")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        baseline_samples[:, 1].numpy(),
        bins=50,
        alpha=0.5,
        label="Baseline HMC",
        density=True,
    )
    axes[1].hist(
        pinn_samples[:, 1].numpy(),
        bins=50,
        alpha=0.5,
        label="PINN-HMC",
        density=True,
    )
    axes[1].set_title(f"Marginal of q2 ({method})")
    axes[1].set_xlabel("q2")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    marginals_path = save_dir / f"{method}_marginals.png"
    plt.savefig(marginals_path, dpi=200)
    plt.close()
    print(f">>> Saved marginal histograms: {marginals_path.resolve()}")

    # --------------------------------------------------
    # 3. Summary text file
    # --------------------------------------------------
    baseline_mean = baseline_samples.mean(dim=0)
    baseline_std = baseline_samples.std(dim=0)
    pinn_mean = pinn_samples.mean(dim=0)
    pinn_std = pinn_samples.std(dim=0)

    baseline_stats = ckpt.get("baseline_stats", {})
    pinn_stats = ckpt.get("pinn_stats", {})
    trajectory_diag = ckpt.get("trajectory_diag", {})

    summary_lines = [
        f"Method: {method}",
        f"Checkpoint: {checkpoint_path}",
        "",
        "=== Sample Statistics ===",
        f"Baseline mean: {baseline_mean.tolist()}",
        f"Baseline std : {baseline_std.tolist()}",
        f"PINN mean    : {pinn_mean.tolist()}",
        f"PINN std     : {pinn_std.tolist()}",
        "",
        "=== Baseline HMC Stats ===",
        f"{baseline_stats}",
        "",
        "=== PINN-HMC Stats ===",
        f"{pinn_stats}",
        "",
        "=== Trajectory Diagnostic ===",
        f"{trajectory_diag}",
    ]

    summary_path = save_dir / f"{method}_sampling_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f">>> Saved summary text: {summary_path.resolve()}")

    print(">>> Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file, e.g. checkpoint_fixed.pt",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional directory to save plots",
    )
    args = parser.parse_args()

    plot_sampling(args.checkpoint, args.save_dir)