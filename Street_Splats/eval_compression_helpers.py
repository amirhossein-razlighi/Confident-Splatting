import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path

from confidence_utils import confidence_values

###############################################################################
#          Helper for evaluating how well our compression is                  #
###############################################################################

def _render_dataset(runner, splats_override, stage="val"):
    """Run the usual validation loop with an arbitrary splat dict.

    Returns average (PSNR, SSIM, LPIPS) over the dataset.
    This is factored out so we can call it many times with different
    confidence-thresholded splat sets without copy-pasting the whole loop.
    """
    device = runner.device
    cfg    = runner.cfg
    valloader = torch.utils.data.DataLoader(
        runner.valset, batch_size=1, shuffle=False, num_workers=1
    )
    metrics = defaultdict(list)

    for data in valloader:
        camtoworlds = data["camtoworld"].to(device)
        Ks          = data["K"].to(device)
        pixels      = data["image"].to(device) / 255.0
        masks       = data.get("mask", None)
        if masks is not None:
            masks = masks.to(device)
        H, W = pixels.shape[1:3]

        colors, _, _ = runner.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=W,
            height=H,
            sh_degree=cfg.sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            masks=masks,
            splats_override=splats_override,
        )
        # clip to [0,1] and convert to CHW for metrics
        colors = torch.clamp(colors, 0.0, 1.0)
        colors_p = colors.permute(0, 3, 1, 2)
        pixels_p = pixels.permute(0, 3, 1, 2)

        metrics["psnr"].append(runner.psnr(colors_p, pixels_p))
        metrics["ssim"].append(runner.ssim(colors_p, pixels_p))
        metrics["lpips"].append(runner.lpips(colors_p, pixels_p))

    # stack → scalar
    out = {
        k: torch.stack(v).mean().item() for k, v in metrics.items()
    }
    return out


def evaluate_compression_curve(runner, thresholds=None, save_dir="plots", stage="val"):
    """Evaluate quality vs. #splats for a list of confidence thresholds.

    * runner … your existing Runner instance (already loaded with ckpt)
    * thresholds … iterable of float thresholds.  Defaults to 0→0.9 step 0.05
    * save_dir … where to dump .png & .csv
    Returns the pandas DataFrame with all statistics.
    """
    cfg = runner.cfg
    device = runner.device
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # default thresholds
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.9, 19)  # 0,0.05,…,0.9

    # pre‑compute confidences once on CPU
    conf = confidence_values(runner.splats).cpu()

    rows = []
    for thr in thresholds:
        keep = conf > thr
        n_keep = int(keep.sum())
        frac   = n_keep / len(conf)

        # build a *detached* splat dict to avoid autograd/comms overhead
        splats_thr = {k: v.detach()[keep.to(v.device)] for k, v in runner.splats.items()}

        stats = _render_dataset(runner, splats_thr, stage=stage)
        rows.append({
            "threshold": thr,
            "num_splats": n_keep,
            "fraction": frac,
            **stats,
        })
        print(f"thr={thr:.2f}  num={n_keep}  PSNR={stats['psnr']:.2f}  SSIM={stats['ssim']:.3f}  LPIPS={stats['lpips']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(Path(save_dir)/f"compression_curve_{stage}.csv", index=False)

    # ── plotting ─────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax2 = ax1.twinx()

    ax1.plot(df["threshold"], df["psnr"], marker="o")
    ax1.set_xlabel("confidence threshold")
    ax1.set_ylabel("PSNR ↑")

    ax2.plot(df["threshold"], df["num_splats"], marker="s", linestyle="--")
    ax2.set_ylabel("# splats kept ↓")

    ax1.grid(True, which="both", linestyle=":", linewidth=0.5)
    plt.title("Quality / compression trade-off")
    plt.tight_layout()
    out_path = Path(save_dir)/f"compression_curve_{stage}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot → {out_path.absolute()}")
    return df
