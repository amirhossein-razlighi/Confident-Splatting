import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


STEP_RE = re.compile(r"_(?:step)?(\d+)")


def _step_from_path(path: Path) -> int:
    match = STEP_RE.search(path.stem)
    return int(match.group(1)) if match else -1


def _latest_json(stats_dir: Path, prefix: str) -> Path | None:
    files = sorted(stats_dir.glob(f"{prefix}_step*.json"), key=_step_from_path)
    return files[-1] if files else None


def _scene_variant(result_dir: Path, results_root: Path) -> tuple[str, str]:
    rel = result_dir.relative_to(results_root)
    parts = rel.parts
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "unknown", parts[-1]


def collect_final_metrics(results_root: Path) -> pd.DataFrame:
    rows = []
    for stats_dir in results_root.rglob("stats"):
        result_dir = stats_dir.parent
        scene, variant = _scene_variant(result_dir, results_root)
        val_json = _latest_json(stats_dir, "val")
        if val_json is None:
            continue
        compressed_json = _latest_json(stats_dir, "val_compressed")

        row = pd.read_json(val_json, typ="series").to_dict()
        row.update(
            {
                "scene": scene,
                "variant": variant,
                "result_dir": str(result_dir),
                "step": _step_from_path(val_json),
                "stage": "val",
            }
        )
        rows.append(row)

        if compressed_json is not None:
            comp = pd.read_json(compressed_json, typ="series").to_dict()
            comp.update(
                {
                    "scene": scene,
                    "variant": variant,
                    "result_dir": str(result_dir),
                    "step": _step_from_path(compressed_json),
                    "stage": "val_compressed",
                }
            )
            rows.append(comp)
    return pd.DataFrame(rows)


def collect_pruning_curves(results_root: Path) -> pd.DataFrame:
    rows = []
    for csv_path in results_root.rglob("plots/pruning_curve_val.csv"):
        result_dir = csv_path.parents[1]
        scene, variant = _scene_variant(result_dir, results_root)
        df = pd.read_csv(csv_path)
        df["scene"] = scene
        df["variant"] = variant
        df["result_dir"] = str(result_dir)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_ablation_summary(final_df: pd.DataFrame, output_dir: Path) -> None:
    if final_df.empty:
        return
    val = final_df[final_df["stage"] == "val"].copy()
    if val.empty:
        return
    summary = (
        val.groupby("variant", as_index=False)
        .agg({"psnr": "mean", "ssim": "mean", "lpips": "mean", "num_GS": "mean"})
        .sort_values("psnr", ascending=False)
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric, label in zip(
        axes, ["psnr", "ssim", "lpips"], ["PSNR", "SSIM", "LPIPS"]
    ):
        ax.bar(summary["variant"], summary[metric], color="tab:blue")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_metrics.png", dpi=220)
    plt.close(fig)


def plot_budget_summary(pruning_df: pd.DataFrame, output_dir: Path) -> None:
    if pruning_df.empty:
        return
    budget = pruning_df[pruning_df["sweep"] == "budget"].copy()
    if budget.empty:
        return
    grouped = (
        budget.groupby(["variant", "selector", "target_fraction"], as_index=False)
        .agg({"psnr": "mean", "ssim": "mean", "lpips": "mean"})
        .sort_values("target_fraction", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (variant, selector), group in grouped.groupby(["variant", "selector"]):
        label = f"{variant}/{selector}"
        ax.plot(group["target_fraction"], group["psnr"], marker="o", label=label)
    ax.set_xlabel("Retained splat fraction")
    ax.set_ylabel("Mean PSNR")
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "budget_psnr_summary.png", dpi=220)
    plt.close(fig)


def write_correlations(pruning_df: pd.DataFrame, output_dir: Path) -> None:
    if pruning_df.empty:
        return
    cols = [
        "avg_conf_kept",
        "avg_conf_all",
        "avg_opacity_kept",
        "fraction",
        "num_splats",
        "psnr",
        "ssim",
        "lpips",
    ]
    available = [c for c in cols if c in pruning_df.columns]
    df = pruning_df[available].dropna(axis=1, how="all")
    if df.shape[1] < 2:
        return
    df.corr(method="pearson").to_csv(output_dir / "all_pruning_pearson.csv")
    df.corr(method="spearman").to_csv(output_dir / "all_pruning_spearman.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_df = collect_final_metrics(args.results_root)
    pruning_df = collect_pruning_curves(args.results_root)

    if not final_df.empty:
        final_df.to_csv(args.output_dir / "final_metrics_long.csv", index=False)
        summary = (
            final_df.groupby(["stage", "variant"], as_index=False)
            .agg(
                {
                    "psnr": ["mean", "std"],
                    "ssim": ["mean", "std"],
                    "lpips": ["mean", "std"],
                    "num_GS": ["mean", "std"],
                    "splats_mem_GB": ["mean", "std"],
                    "ellipse_time": ["mean", "std"],
                }
            )
        )
        summary.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns
        ]
        summary.to_csv(args.output_dir / "final_metrics_summary.csv", index=False)
        plot_ablation_summary(final_df, args.output_dir)

    if not pruning_df.empty:
        pruning_df.to_csv(args.output_dir / "pruning_curves_long.csv", index=False)
        budget_summary = (
            pruning_df[pruning_df["sweep"] == "budget"]
            .groupby(["variant", "selector", "target_fraction"], as_index=False)
            .agg(
                {
                    "psnr": ["mean", "std"],
                    "ssim": ["mean", "std"],
                    "lpips": ["mean", "std"],
                    "fraction": ["mean", "std"],
                    "compression_ratio": ["mean", "std"],
                }
            )
        )
        budget_summary.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in budget_summary.columns
        ]
        budget_summary.to_csv(args.output_dir / "budget_summary.csv", index=False)
        plot_budget_summary(pruning_df, args.output_dir)
        write_correlations(pruning_df, args.output_dir)

    print(f"Wrote aggregate review artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
