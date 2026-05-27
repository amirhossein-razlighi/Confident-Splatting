import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from confidence_utils import confidence_values


def _sync_if_needed(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _memory_breakdown_mb(splats: dict) -> dict:
    confidence_keys = {"conf_alpha", "conf_beta", "conf_logit"}
    total = sum(_tensor_bytes(v) for v in splats.values() if torch.is_tensor(v))
    confidence = sum(
        _tensor_bytes(v)
        for k, v in splats.items()
        if k in confidence_keys and torch.is_tensor(v)
    )
    return {
        "splat_mem_mb": total / (1024**2),
        "confidence_mem_mb": confidence / (1024**2),
        "render_mem_mb": (total - confidence) / (1024**2),
    }


def _mask_splats(splats: dict, keep: torch.Tensor) -> dict:
    keep = keep.bool()
    n = keep.numel()
    out = {}
    for key, value in splats.items():
        if torch.is_tensor(value) and value.shape[:1] == (n,):
            out[key] = value.detach()[keep.to(value.device)]
        elif torch.is_tensor(value):
            out[key] = value.detach()
    return out


def _selector_scores(runner, selector: str, seed: int = 0) -> torch.Tensor:
    splats = runner.splats
    device = runner.device
    if selector == "confidence":
        return confidence_values(splats).detach().to(device)
    if selector == "opacity":
        return torch.sigmoid(splats["opacities"]).detach().to(device)
    if selector == "random":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return torch.rand(splats["means"].shape[0], generator=generator, device=device)
    raise ValueError(f"Unknown pruning selector: {selector}")


def _top_fraction_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    n_total = scores.numel()
    n_keep = max(1, min(n_total, int(round(n_total * float(fraction)))))
    keep = torch.zeros(n_total, dtype=torch.bool, device=scores.device)
    keep[torch.topk(scores, n_keep).indices] = True
    return keep


def _threshold_mask(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    keep = scores > float(threshold)
    if not torch.any(keep):
        keep[torch.argmax(scores)] = True
    return keep


def _render_dataset(runner, splats_override, stage: str = "val") -> dict:
    """Render validation images with an arbitrary splat dictionary."""
    device = runner.device
    cfg = runner.cfg
    valloader = DataLoader(runner.valset, batch_size=1, shuffle=False, num_workers=1)
    metrics = defaultdict(list)
    elapsed = 0.0

    for data in valloader:
        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        masks = data.get("mask", None)
        if masks is not None:
            masks = masks.to(device)
        height, width = pixels.shape[1:3]

        _sync_if_needed(device)
        tic = time.time()
        colors, _, _ = runner.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=cfg.sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            masks=masks,
            splats_override=splats_override,
            eval_mode=True,
        )
        _sync_if_needed(device)
        elapsed += time.time() - tic

        colors = torch.clamp(colors, 0.0, 1.0)
        colors_p = colors.permute(0, 3, 1, 2)
        pixels_p = pixels.permute(0, 3, 1, 2)

        metrics["psnr"].append(runner.psnr(colors_p, pixels_p))
        metrics["ssim"].append(runner.ssim(colors_p, pixels_p))
        metrics["lpips"].append(runner.lpips(colors_p, pixels_p))

    out = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
    out["render_time_s"] = elapsed / max(1, len(valloader))
    return out


def _first_validation_pair(runner, splats_override) -> tuple[np.ndarray, np.ndarray]:
    device = runner.device
    cfg = runner.cfg
    loader = DataLoader(runner.valset, batch_size=1, shuffle=False, num_workers=1)
    data = next(iter(loader))
    camtoworlds = data["camtoworld"].to(device)
    Ks = data["K"].to(device)
    pixels = data["image"].to(device) / 255.0
    masks = data.get("mask", None)
    if masks is not None:
        masks = masks.to(device)
    height, width = pixels.shape[1:3]

    colors, _, _ = runner.rasterize_splats(
        camtoworlds=camtoworlds,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=cfg.sh_degree,
        near_plane=cfg.near_plane,
        far_plane=cfg.far_plane,
        masks=masks,
        splats_override=splats_override,
        eval_mode=True,
    )
    gt = torch.clamp(pixels, 0.0, 1.0)[0].cpu().numpy()
    pred = torch.clamp(colors, 0.0, 1.0)[0].cpu().detach().numpy()
    return (gt * 255).astype(np.uint8), (pred * 255).astype(np.uint8)


def _row_for_mask(
    runner,
    selector: str,
    sweep: str,
    keep: torch.Tensor,
    base_stats: dict,
    threshold: Optional[float] = None,
    target_fraction: Optional[float] = None,
    trial: int = 0,
    stage: str = "val",
) -> dict:
    splats_pruned = _mask_splats(runner.splats, keep)
    stats = _render_dataset(runner, splats_pruned, stage=stage)
    n_total = runner.splats["means"].shape[0]
    n_keep = splats_pruned["means"].shape[0]

    conf = None
    if runner.cfg.use_conf_scores and (
        "conf_alpha" in runner.splats or "conf_logit" in runner.splats
    ):
        conf = confidence_values(runner.splats).detach().cpu()
    opacity = torch.sigmoid(runner.splats["opacities"]).detach().cpu()
    keep_cpu = keep.detach().cpu()

    row = {
        "stage": stage,
        "selector": selector,
        "sweep": sweep,
        "threshold": threshold,
        "target_fraction": target_fraction,
        "trial": trial,
        "num_splats": n_keep,
        "total_splats": n_total,
        "fraction": n_keep / n_total,
        "compression_ratio": n_total / max(1, n_keep),
        "avg_opacity_kept": opacity[keep_cpu].mean().item(),
        "avg_opacity_all": opacity.mean().item(),
        **_memory_breakdown_mb(splats_pruned),
        **stats,
    }
    if conf is not None:
        row["avg_conf_kept"] = conf[keep_cpu].mean().item()
        row["avg_conf_all"] = conf.mean().item()
    else:
        row["avg_conf_kept"] = np.nan
        row["avg_conf_all"] = np.nan

    for metric in ("psnr", "ssim", "lpips"):
        row[f"{metric}_base"] = base_stats[metric]
    row["psnr_drop"] = base_stats["psnr"] - row["psnr"]
    row["ssim_drop"] = base_stats["ssim"] - row["ssim"]
    row["lpips_delta"] = row["lpips"] - base_stats["lpips"]
    return row


def evaluate_pruning_curves(
    runner,
    thresholds: Optional[Iterable[float]] = None,
    keep_fractions: Optional[Iterable[float]] = None,
    selectors: Optional[Iterable[str]] = None,
    random_trials: int = 3,
    save_dir: str = "plots",
    stage: str = "val",
) -> pd.DataFrame:
    """Evaluate threshold and matched-budget pruning curves.

    Threshold curves answer whether the chosen confidence cutoff is stable.
    Matched-budget curves answer whether confidence is a better pruning score
    than simpler post-hoc scores at the same retained-splat budget.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    thresholds = list(thresholds if thresholds is not None else np.linspace(0.0, 0.9, 19))
    keep_fractions = list(
        keep_fractions if keep_fractions is not None else [1.0, 0.75, 0.5, 0.25, 0.1]
    )
    selectors = list(selectors if selectors is not None else ["confidence", "opacity", "random"])

    if not runner.cfg.use_conf_scores:
        selectors = [s for s in selectors if s != "confidence"]

    base_splats = {k: v.detach() for k, v in runner.splats.items() if torch.is_tensor(v)}
    base_stats = _render_dataset(runner, base_splats, stage=stage)
    rows = []

    for selector in selectors:
        trials = max(1, random_trials if selector == "random" else 1)
        for trial in range(trials):
            scores = _selector_scores(runner, selector, seed=trial)

            if selector in {"confidence", "opacity"}:
                for threshold in thresholds:
                    keep = _threshold_mask(scores, threshold)
                    rows.append(
                        _row_for_mask(
                            runner,
                            selector=selector,
                            sweep="threshold",
                            keep=keep,
                            base_stats=base_stats,
                            threshold=float(threshold),
                            trial=trial,
                            stage=stage,
                        )
                    )

            for fraction in keep_fractions:
                keep = _top_fraction_mask(scores, fraction)
                rows.append(
                    _row_for_mask(
                        runner,
                        selector=selector,
                        sweep="budget",
                        keep=keep,
                        base_stats=base_stats,
                        target_fraction=float(fraction),
                        trial=trial,
                        stage=stage,
                    )
                )

    df = pd.DataFrame(rows)
    csv_path = save_path / f"pruning_curve_{stage}.csv"
    json_path = save_path / f"pruning_curve_{stage}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"base": base_stats, "rows": rows}, indent=2))
    print(f"Saved pruning CSV to {csv_path.absolute()}")
    print(f"Saved pruning JSON to {json_path.absolute()}")

    legacy = df[(df["selector"] == "confidence") & (df["sweep"] == "threshold")].copy()
    if not legacy.empty:
        legacy.to_csv(save_path / f"compression_curve_{stage}.csv", index=False)

    _plot_budget_curve(df, save_path, stage)
    _plot_threshold_sensitivity(df, save_path, stage)
    _plot_quality_size_pareto(df, save_path, stage)
    _plot_qualitative_budget_grid(runner, df, save_path, stage)
    _write_correlation_report(df, save_path, stage)
    return df


def _plot_budget_curve(df: pd.DataFrame, save_path: Path, stage: str) -> None:
    budget = df[df["sweep"] == "budget"].copy()
    if budget.empty:
        return
    grouped = (
        budget.groupby(["selector", "target_fraction"], as_index=False)
        .agg({"psnr": "mean", "ssim": "mean", "lpips": "mean", "num_splats": "mean"})
        .sort_values("target_fraction", ascending=False)
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    metrics = [("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS")]
    for ax, (metric, label) in zip(axes, metrics):
        for selector, group in grouped.groupby("selector"):
            ax.plot(group["target_fraction"], group[metric], marker="o", label=selector)
        ax.set_xlabel("Retained splat fraction")
        ax.set_ylabel(label)
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.invert_xaxis()
    axes[0].legend(frameon=False)
    fig.tight_layout()
    out = save_path / f"budget_curve_{stage}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved budget curve to {out.absolute()}")


def _plot_threshold_sensitivity(df: pd.DataFrame, save_path: Path, stage: str) -> None:
    threshold = df[(df["sweep"] == "threshold") & (df["selector"] == "confidence")]
    if threshold.empty:
        return

    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    ax2 = ax1.twinx()
    ax1.plot(threshold["threshold"], threshold["psnr"], marker="o", label="PSNR")
    ax1.plot(threshold["threshold"], threshold["ssim"], marker="^", label="SSIM")
    ax2.plot(
        threshold["threshold"],
        threshold["fraction"],
        marker="s",
        linestyle="--",
        color="tab:green",
        label="retained fraction",
    )
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Quality")
    ax2.set_ylabel("Retained splat fraction")
    ax1.grid(True, linestyle=":", linewidth=0.6)
    ax1.legend(loc="lower left", frameon=False)
    ax2.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    out = save_path / f"threshold_sensitivity_{stage}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved threshold sensitivity plot to {out.absolute()}")


def _plot_quality_size_pareto(df: pd.DataFrame, save_path: Path, stage: str) -> None:
    budget = df[df["sweep"] == "budget"].copy()
    if budget.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for selector, group in budget.groupby("selector"):
        group = group.groupby("fraction", as_index=False).agg({"psnr": "mean", "lpips": "mean"})
        ax.plot(group["fraction"], group["psnr"], marker="o", label=selector)
    ax.set_xlabel("Retained splat fraction")
    ax.set_ylabel("PSNR")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.invert_xaxis()
    ax.legend(frameon=False)
    fig.tight_layout()
    out = save_path / f"quality_size_pareto_{stage}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved Pareto plot to {out.absolute()}")


def _plot_qualitative_budget_grid(runner, df: pd.DataFrame, save_path: Path, stage: str) -> None:
    budget = df[df["sweep"] == "budget"].copy()
    if budget.empty:
        return

    fractions = sorted(budget["target_fraction"].dropna().unique(), reverse=True)
    if len(fractions) > 3:
        fractions = [fractions[0], fractions[len(fractions) // 2], fractions[-1]]
    selectors = list(budget["selector"].unique())

    first_gt = None
    rows = len(selectors)
    cols = len(fractions) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axes = np.atleast_2d(axes)

    for r, selector in enumerate(selectors):
        scores = _selector_scores(runner, selector, seed=0)
        for c, fraction in enumerate([None] + fractions):
            ax = axes[r, c]
            if fraction is None:
                splats = {k: v.detach() for k, v in runner.splats.items() if torch.is_tensor(v)}
                gt, pred = _first_validation_pair(runner, splats)
                first_gt = gt if first_gt is None else first_gt
                image = first_gt
                title = "GT"
            else:
                keep = _top_fraction_mask(scores, float(fraction))
                splats = _mask_splats(runner.splats, keep)
                _, image = _first_validation_pair(runner, splats)
                title = f"{selector}, {fraction:.0%}"
            ax.imshow(image)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.tight_layout()
    out = save_path / f"qualitative_budget_grid_{stage}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved qualitative grid to {out.absolute()}")


def _write_correlation_report(df: pd.DataFrame, save_path: Path, stage: str) -> None:
    cols = [
        "avg_conf_kept",
        "avg_conf_all",
        "avg_opacity_kept",
        "fraction",
        "num_splats",
        "splat_mem_mb",
        "psnr",
        "ssim",
        "lpips",
    ]
    available = [c for c in cols if c in df.columns]
    corr_input = df[available].dropna(axis=1, how="all")
    if corr_input.shape[1] < 2:
        return
    pearson = corr_input.corr(method="pearson")
    spearman = corr_input.corr(method="spearman")
    pearson.to_csv(save_path / f"correlation_pearson_{stage}.csv")
    spearman.to_csv(save_path / f"correlation_spearman_{stage}.csv")


def evaluate_compression_curve(runner, thresholds=None, save_dir="plots", stage="val"):
    """Legacy confidence-threshold curve kept for existing paper scripts."""
    if not runner.cfg.use_conf_scores:
        print("Skipping confidence threshold curve because use_conf_scores is false.")
        return pd.DataFrame()

    df = evaluate_pruning_curves(
        runner,
        thresholds=thresholds,
        keep_fractions=[],
        selectors=["confidence"],
        random_trials=1,
        save_dir=save_dir,
        stage=stage,
    )
    threshold_df = df[(df["selector"] == "confidence") & (df["sweep"] == "threshold")].copy()
    out_csv = Path(save_dir) / f"compression_curve_{stage}.csv"
    threshold_df.to_csv(out_csv, index=False)

    if not threshold_df.empty:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()
        ax1.plot(threshold_df["threshold"], threshold_df["psnr"], marker="o", label="PSNR")
        ax2.plot(
            threshold_df["threshold"],
            threshold_df["num_splats"],
            marker="s",
            linestyle="--",
            color="tab:green",
            label="splats",
        )
        ax1.set_xlabel("Confidence threshold")
        ax1.set_ylabel("PSNR")
        ax2.set_ylabel("Splats kept")
        ax1.grid(True, linestyle=":", linewidth=0.6)
        fig.tight_layout()
        out_png = Path(save_dir) / f"compression_curve_{stage}.png"
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        print(f"Saved legacy compression curve to {out_png.absolute()}")
    return threshold_df
