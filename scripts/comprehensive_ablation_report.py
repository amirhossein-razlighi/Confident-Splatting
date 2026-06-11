import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F


STEP_RE = re.compile(r"_(?:step)?(\d+)")


def _step_from_path(path: Path) -> int:
    match = STEP_RE.search(path.stem)
    return int(match.group(1)) if match else -1


def _latest_json(stats_dir: Path, prefix: str) -> Path | None:
    files = sorted(stats_dir.glob(f"{prefix}_step*.json"), key=_step_from_path)
    return files[-1] if files else None


def _latest_ckpt(ckpt_dir: Path) -> Path | None:
    files = sorted(ckpt_dir.glob("ckpt_*_rank0.pt"), key=_step_from_path)
    return files[-1] if files else None


def _read_manifest(path: Path, scene: str | None, variant_regex: str | None) -> pd.DataFrame:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if scene and row["scene"] != scene:
                continue
            if variant_regex and not re.search(variant_regex, row["variant"]):
                continue
            rows.append(row)
    return pd.DataFrame(rows)


def _load_final_metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest.to_dict("records"):
        stats_dir = Path(row["result_dir"]) / "stats"
        val_json = _latest_json(stats_dir, "val")
        if val_json is None:
            continue
        metrics = pd.read_json(val_json, typ="series").to_dict()
        metrics.update(
            {
                "scene": row["scene"],
                "variant": row["variant"],
                "config": row["config"],
                "result_dir": row["result_dir"],
                "step": _step_from_path(val_json),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _load_pruning_curves(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest.to_dict("records"):
        csv_path = Path(row["result_dir"]) / "plots" / "pruning_curve_val.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["scene"] = row["scene"]
        df["variant"] = row["variant"]
        df["config"] = row["config"]
        df["result_dir"] = row["result_dir"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _confidence_tensor(splats: dict) -> tuple[torch.Tensor | None, float]:
    if "conf_logit" in splats:
        conf = torch.sigmoid(splats["conf_logit"]).squeeze(-1)
        return conf, math.nan
    if "conf_alpha" in splats and "conf_beta" in splats:
        alpha = F.softplus(splats["conf_alpha"]) + 1e-6
        beta = F.softplus(splats["conf_beta"]) + 1e-6
        conf = (alpha / (alpha + beta)).squeeze(-1)
        entropy = torch.distributions.Beta(alpha, beta).entropy().mean().item()
        return conf, entropy
    return None, math.nan


def _gini(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    x = np.sort(values.astype(np.float64))
    total = x.sum()
    if total <= 0:
        return 0.0
    n = x.size
    return float((2.0 * np.arange(1, n + 1).dot(x)) / (n * total) - (n + 1) / n)


def _load_confidence_stats(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    thresholds = [0.05, 0.10, 0.25, 0.50]
    for row in manifest.to_dict("records"):
        result_dir = Path(row["result_dir"])
        ckpt = _latest_ckpt(result_dir / "ckpts")
        out = {
            "scene": row["scene"],
            "variant": row["variant"],
            "config": row["config"],
            "result_dir": row["result_dir"],
            "ckpt": str(ckpt) if ckpt else "",
        }
        if ckpt is None:
            rows.append(out)
            continue
        try:
            data = torch.load(ckpt, map_location="cpu", weights_only=True)
            conf, beta_entropy = _confidence_tensor(data.get("splats", {}))
        except Exception as exc:
            out["error"] = repr(exc)
            rows.append(out)
            continue
        if conf is None:
            rows.append(out)
            continue

        conf_np = conf.detach().float().cpu().numpy()
        out.update(
            {
                "conf_count": int(conf_np.size),
                "conf_mean": float(np.mean(conf_np)),
                "conf_std": float(np.std(conf_np)),
                "conf_var": float(np.var(conf_np)),
                "conf_min": float(np.min(conf_np)),
                "conf_max": float(np.max(conf_np)),
                "conf_gini": _gini(conf_np),
                "beta_entropy": beta_entropy,
            }
        )
        for q in quantiles:
            out[f"conf_q{int(q * 100):02d}"] = float(np.quantile(conf_np, q))
        for t in thresholds:
            out[f"frac_conf_lt_{str(t).replace('.', '_')}"] = float(np.mean(conf_np < t))
        rows.append(out)
    return pd.DataFrame(rows)


def _auc_by_fraction(df: pd.DataFrame, metric: str) -> float:
    if df.empty or metric not in df:
        return math.nan
    points = df[["target_fraction", metric]].dropna().drop_duplicates("target_fraction")
    if points.shape[0] < 2:
        return math.nan
    points = points.sort_values("target_fraction")
    span = points["target_fraction"].max() - points["target_fraction"].min()
    if span <= 0:
        return math.nan
    return float(np.trapz(points[metric], points["target_fraction"]) / span)


def _min_fraction_with_drop(df: pd.DataFrame, max_drop: float) -> float:
    if df.empty or "psnr_drop" not in df:
        return math.nan
    ok = df[df["psnr_drop"] <= max_drop]
    if ok.empty:
        return math.nan
    return float(ok["target_fraction"].min())


def _psnr_at_fraction(df: pd.DataFrame, fraction: float) -> float:
    if df.empty:
        return math.nan
    row = df[np.isclose(df["target_fraction"], fraction)]
    if row.empty:
        return math.nan
    return float(row["psnr"].mean())


def _threshold_width(df: pd.DataFrame, max_drop: float) -> float:
    ok = df[df["psnr_drop"] <= max_drop]
    if ok.empty:
        return 0.0
    return float(ok["threshold"].max() - ok["threshold"].min())


def _build_summary(final_df: pd.DataFrame, pruning_df: pd.DataFrame, conf_df: pd.DataFrame) -> pd.DataFrame:
    variants = sorted(set(final_df.get("variant", [])) | set(pruning_df.get("variant", [])))
    rows = []
    for variant in variants:
        final = final_df[final_df["variant"] == variant]
        budget_conf = pruning_df[
            (pruning_df["variant"] == variant)
            & (pruning_df["sweep"] == "budget")
            & (pruning_df["selector"] == "confidence")
        ]
        threshold_conf = pruning_df[
            (pruning_df["variant"] == variant)
            & (pruning_df["sweep"] == "threshold")
            & (pruning_df["selector"] == "confidence")
        ]
        row = {"variant": variant}
        if not final.empty:
            first = final.iloc[0]
            for col in ["scene", "config", "psnr", "ssim", "lpips", "num_GS", "render_mem_GB"]:
                if col in first:
                    row[f"final_{col}"] = first[col]
        row.update(
            {
                "auc_psnr_confidence": _auc_by_fraction(budget_conf, "psnr"),
                "auc_ssim_confidence": _auc_by_fraction(budget_conf, "ssim"),
                "auc_lpips_confidence_lower_better": _auc_by_fraction(budget_conf, "lpips"),
                "min_keep_frac_psnr_drop_le_0_10": _min_fraction_with_drop(budget_conf, 0.10),
                "min_keep_frac_psnr_drop_le_0_25": _min_fraction_with_drop(budget_conf, 0.25),
                "min_keep_frac_psnr_drop_le_0_50": _min_fraction_with_drop(budget_conf, 0.50),
                "psnr_at_keep_25": _psnr_at_fraction(budget_conf, 0.25),
                "psnr_at_keep_15": _psnr_at_fraction(budget_conf, 0.15),
                "psnr_at_keep_10": _psnr_at_fraction(budget_conf, 0.10),
                "psnr_at_keep_05": _psnr_at_fraction(budget_conf, 0.05),
                "threshold_width_drop_le_0_25": _threshold_width(threshold_conf, 0.25),
                "threshold_width_drop_le_0_50": _threshold_width(threshold_conf, 0.50),
            }
        )
        conf = conf_df[conf_df["variant"] == variant]
        if not conf.empty:
            for col in [
                "conf_mean",
                "conf_std",
                "conf_var",
                "conf_gini",
                "conf_q05",
                "conf_q50",
                "conf_q95",
                "frac_conf_lt_0_05",
                "frac_conf_lt_0_1",
                "frac_conf_lt_0_25",
                "frac_conf_lt_0_5",
                "beta_entropy",
            ]:
                if col in conf:
                    row[col] = conf.iloc[0].get(col, math.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_selector_gain(pruning_df: pd.DataFrame) -> pd.DataFrame:
    budget = pruning_df[pruning_df["sweep"] == "budget"].copy()
    if budget.empty:
        return pd.DataFrame()
    grouped = (
        budget.groupby(["scene", "variant", "selector", "target_fraction"], as_index=False)
        .agg({"psnr": "mean", "ssim": "mean", "lpips": "mean"})
    )
    rows = []
    for (scene, variant, fraction), group in grouped.groupby(["scene", "variant", "target_fraction"]):
        by_selector = group.set_index("selector")
        if "confidence" not in by_selector.index:
            continue
        row = {
            "scene": scene,
            "variant": variant,
            "target_fraction": fraction,
            "confidence_psnr": by_selector.loc["confidence", "psnr"],
        }
        for selector in ["opacity", "random"]:
            if selector in by_selector.index:
                row[f"{selector}_psnr"] = by_selector.loc[selector, "psnr"]
                row[f"confidence_minus_{selector}_psnr"] = (
                    by_selector.loc["confidence", "psnr"] - by_selector.loc[selector, "psnr"]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("No rows.\n")
        return
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    table = table.fillna("").astype(str)
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def _plot_confidence_budget(pruning_df: pd.DataFrame, output_dir: Path) -> None:
    budget = pruning_df[
        (pruning_df["sweep"] == "budget") & (pruning_df["selector"] == "confidence")
    ].copy()
    if budget.empty:
        return
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    sns.lineplot(
        data=budget,
        x="target_fraction",
        y="psnr",
        hue="variant",
        marker="o",
        linewidth=2.0,
        ax=ax,
    )
    ax.invert_xaxis()
    ax.set_xlabel("Retained splat fraction")
    ax.set_ylabel("PSNR")
    ax.set_title("Confidence-pruned quality at matched budgets")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_budget_psnr.png", dpi=260)
    plt.close(fig)


def _plot_selector_gain(gain_df: pd.DataFrame, output_dir: Path) -> None:
    if gain_df.empty or "confidence_minus_opacity_psnr" not in gain_df:
        return
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    sns.lineplot(
        data=gain_df,
        x="target_fraction",
        y="confidence_minus_opacity_psnr",
        hue="variant",
        marker="o",
        linewidth=2.0,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.invert_xaxis()
    ax.set_xlabel("Retained splat fraction")
    ax.set_ylabel("PSNR gain over opacity pruning")
    ax.set_title("Does confidence rank splats better than opacity?")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_minus_opacity_gain.png", dpi=260)
    plt.close(fig)


def _plot_threshold_stability(pruning_df: pd.DataFrame, output_dir: Path) -> None:
    threshold = pruning_df[
        (pruning_df["sweep"] == "threshold") & (pruning_df["selector"] == "confidence")
    ].copy()
    if threshold.empty:
        return
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    sns.lineplot(
        data=threshold,
        x="threshold",
        y="psnr_drop",
        hue="variant",
        marker="o",
        linewidth=2.0,
        ax=axes[0],
    )
    axes[0].axhline(0.25, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    axes[0].axhline(0.50, color="black", linewidth=1.0, linestyle=":", alpha=0.6)
    axes[0].set_title("Quality loss under confidence thresholds")
    axes[0].set_xlabel("Confidence threshold")
    axes[0].set_ylabel("PSNR drop")

    sns.lineplot(
        data=threshold,
        x="threshold",
        y="fraction",
        hue="variant",
        marker="o",
        linewidth=2.0,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Retained fraction under thresholds")
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("Retained splat fraction")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_stability.png", dpi=260)
    plt.close(fig)


def _plot_confidence_stats(conf_df: pd.DataFrame, output_dir: Path) -> None:
    if conf_df.empty or "conf_mean" not in conf_df:
        return
    plot_df = conf_df.dropna(subset=["conf_mean"]).copy()
    if plot_df.empty:
        return
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    order = plot_df.sort_values("conf_mean")["variant"]
    axes[0].bar(
        plot_df.set_index("variant").loc[order].index,
        plot_df.set_index("variant").loc[order]["conf_mean"],
        yerr=plot_df.set_index("variant").loc[order]["conf_std"],
        color=sns.color_palette("deep", len(order)),
        capsize=3,
    )
    axes[0].set_title("Confidence mean and spread")
    axes[0].set_ylabel("Confidence")
    axes[0].tick_params(axis="x", rotation=30)

    low_cols = [c for c in ["frac_conf_lt_0_05", "frac_conf_lt_0_1", "frac_conf_lt_0_25", "frac_conf_lt_0_5"] if c in plot_df]
    low = plot_df[["variant"] + low_cols].melt("variant", var_name="threshold", value_name="fraction")
    low["threshold"] = low["threshold"].str.replace("frac_conf_lt_", "<", regex=False).str.replace("_", ".")
    sns.barplot(data=low, x="variant", y="fraction", hue="threshold", ax=axes[1])
    axes[1].set_title("How much mass is low-confidence?")
    axes[1].set_ylabel("Fraction of splats")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_distribution_summary.png", dpi=260)
    plt.close(fig)


def _plot_budget_heatmap(pruning_df: pd.DataFrame, output_dir: Path) -> None:
    budget = pruning_df[
        (pruning_df["sweep"] == "budget") & (pruning_df["selector"] == "confidence")
    ].copy()
    if budget.empty:
        return
    pivot = budget.pivot_table(index="variant", columns="target_fraction", values="psnr", aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(8.8, max(3.2, 0.45 * len(pivot) + 1.2)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", linewidths=0.4, ax=ax)
    ax.set_xlabel("Retained splat fraction")
    ax.set_ylabel("")
    ax.set_title("PSNR at matched confidence-pruning budgets")
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_budget_psnr_heatmap.png", dpi=260)
    plt.close(fig)


def _write_report(output_dir: Path, manifest: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Comprehensive Ablation Report",
        "",
        f"Runs included: {len(manifest)}",
        "",
        "Key files:",
        "",
        "- `comprehensive_ablation_summary.csv`: one-row-per-variant metric summary",
        "- `comprehensive_ablation_summary.md`: markdown version for quick paste",
        "- `confidence_stats.csv`: checkpoint-derived confidence distribution stats",
        "- `budget_confidence_rows.csv`: matched-budget confidence pruning rows",
        "- `selector_gain_table.csv`: confidence-vs-opacity/random PSNR deltas",
        "- `threshold_confidence_rows.csv`: threshold sensitivity rows",
        "",
        "Plots:",
        "",
        "- `confidence_budget_psnr.png`",
        "- `confidence_budget_psnr_heatmap.png`",
        "- `confidence_minus_opacity_gain.png`",
        "- `threshold_stability.png`",
        "- `confidence_distribution_summary.png`",
        "",
    ]
    if not summary.empty and "auc_psnr_confidence" in summary:
        cols = ["variant", "auc_psnr_confidence", "psnr_at_keep_25", "psnr_at_keep_10", "conf_mean", "conf_std"]
        cols = [c for c in cols if c in summary]
        quick = summary[cols].copy()
        for col in quick.columns:
            if pd.api.types.is_float_dtype(quick[col]):
                quick[col] = quick[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        quick = quick.fillna("").astype(str)
        md = [
            "| " + " | ".join(quick.columns) + " |",
            "| " + " | ".join(["---"] * len(quick.columns)) + " |",
        ]
        for _, row in quick.iterrows():
            md.append("| " + " | ".join(row[c] for c in quick.columns) + " |")
        lines += ["## Quick Summary", "", "\n".join(md), ""]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--variant-regex", type=str, default=None)
    parser.add_argument(
        "--skip-checkpoint-confidence",
        action="store_true",
        help="Skip CPU checkpoint reads for confidence quantiles and variance.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(args.manifest, args.scene, args.variant_regex)
    if manifest.empty:
        raise SystemExit("No manifest rows matched the requested filters.")

    final_df = _load_final_metrics(manifest)
    pruning_df = _load_pruning_curves(manifest)
    conf_df = pd.DataFrame() if args.skip_checkpoint_confidence else _load_confidence_stats(manifest)
    summary = _build_summary(final_df, pruning_df, conf_df)
    selector_gain = _build_selector_gain(pruning_df)

    manifest.to_csv(args.output_dir / "manifest_rows.csv", index=False)
    final_df.to_csv(args.output_dir / "final_metrics.csv", index=False)
    pruning_df.to_csv(args.output_dir / "pruning_curves_long.csv", index=False)
    conf_df.to_csv(args.output_dir / "confidence_stats.csv", index=False)
    summary.to_csv(args.output_dir / "comprehensive_ablation_summary.csv", index=False)
    _write_markdown_table(summary, args.output_dir / "comprehensive_ablation_summary.md")

    budget_conf = pruning_df[
        (pruning_df["sweep"] == "budget") & (pruning_df["selector"] == "confidence")
    ].copy()
    threshold_conf = pruning_df[
        (pruning_df["sweep"] == "threshold") & (pruning_df["selector"] == "confidence")
    ].copy()
    budget_conf.to_csv(args.output_dir / "budget_confidence_rows.csv", index=False)
    threshold_conf.to_csv(args.output_dir / "threshold_confidence_rows.csv", index=False)
    selector_gain.to_csv(args.output_dir / "selector_gain_table.csv", index=False)

    _plot_confidence_budget(pruning_df, args.output_dir)
    _plot_selector_gain(selector_gain, args.output_dir)
    _plot_threshold_stability(pruning_df, args.output_dir)
    _plot_confidence_stats(conf_df, args.output_dir)
    _plot_budget_heatmap(pruning_df, args.output_dir)
    _write_report(args.output_dir, manifest, summary)

    print(f"Wrote comprehensive ablation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
