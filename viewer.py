import argparse
import math
import os
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
import matplotlib.cm as cm
from datetime import datetime

from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization

from confidence_utils import confidence_values

def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.ckpt is None:
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
        ) = load_test_data(device=device, scene_grid=args.scene_grid)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()

        sh_degree = None
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)

        for _ in tqdm.trange(1):
            render_colors, render_alphas, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats,
                Ks,
                width,
                height,
                render_mode="RGB+D",
                packed=False,
                distributed=world_size > 1,
            )
        C = render_colors.shape[0]
        assert render_colors.shape == (C, height, width, 4)
        assert render_alphas.shape == (C, height, width, 1)
        render_colors.sum().backward()

        render_rgbs = render_colors[..., 0:3]
        render_depths = render_colors[..., 3:4]
        render_depths = render_depths / render_depths.max()

        os.makedirs(args.output_dir, exist_ok=True)
        canvas = (
            torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        imageio.imsave(
            f"{args.output_dir}/render_rank{world_rank}.png",
            (canvas * 255).astype(np.uint8),
        )

    else:
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
        conf_alpha, conf_beta = [], []

        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
            means.append(ckpt["means"])
            quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales.append(torch.exp(ckpt["scales"]))
            opacities.append(torch.sigmoid(ckpt["opacities"]))
            sh0.append(ckpt["sh0"])
            shN.append(ckpt["shN"])
            conf_alpha.append(ckpt["conf_alpha"])
            conf_beta.append(ckpt["conf_beta"])

        means = torch.cat(means, dim=0)
        quats = torch.cat(quats, dim=0)
        scales = torch.cat(scales, dim=0)
        opacities = torch.cat(opacities, dim=0)
        sh0 = torch.cat(sh0, dim=0)
        shN = torch.cat(shN, dim=0)
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
        conf_alpha = torch.cat(conf_alpha, dim=0)
        conf_beta = torch.cat(conf_beta, dim=0)

        conf = confidence_values({"conf_alpha": conf_alpha, "conf_beta": conf_beta})

        @torch.no_grad()
        def update_conf_view(threshold, use_conf_color=True):
            mask = conf >= threshold
            masked_idx = mask.nonzero(as_tuple=True)[0]

            if use_conf_color:
                cmap = cm.get_cmap("viridis")
                conf_rgb_np = cmap(conf[masked_idx].detach().cpu().numpy())[:, :3]
                conf_rgb = torch.tensor(conf_rgb_np, dtype=torch.float32, device=device).unsqueeze(0)
                return (
                    means[masked_idx],
                    quats[masked_idx],
                    scales[masked_idx],
                    opacities[masked_idx],
                    conf_rgb.expand(1, -1, -1),
                )
            else:
                sh0_masked = sh0[masked_idx]
                shN_masked = shN[masked_idx]
                colors_masked = torch.cat([sh0_masked, shN_masked], dim=-2)
                return (
                    means[masked_idx],
                    quats[masked_idx],
                    scales[masked_idx],
                    opacities[masked_idx],
                    colors_masked,
                )

        class ViewerRenderer:
            def __init__(self):
                self.show_confidence = False
                self.threshold = 0.0
                self.render_data = update_conf_view(self.threshold)
                self.last_render = None

            def __call__(self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
                width, height = img_wh
                c2w = camera_state.c2w
                K = camera_state.get_K(img_wh)
                c2w = torch.from_numpy(c2w).float().to(device)
                K = torch.from_numpy(K).float().to(device)
                viewmat = c2w.inverse()

                if args.backend == "gsplat":
                    rasterization_fn = rasterization
                elif args.backend == "inria":
                    from gsplat import rasterization_inria_wrapper
                    rasterization_fn = rasterization_inria_wrapper
                else:
                    raise ValueError

                if self.show_confidence:
                    m, q, s, o, c = self.render_data
                    render_colors, _, _ = rasterization_fn(
                        m, q, s, o, c,
                        viewmat[None], K[None], width, height,
                        render_mode="RGB", radius_clip=3,
                    )
                elif self.threshold > 0.0:
                    m, q, s, o, c = update_conf_view(self.threshold, use_conf_color=False)
                    render_colors, _, _ = rasterization_fn(
                        m, q, s, o, c,
                        viewmat[None], K[None], width, height,
                        sh_degree=sh_degree,
                        render_mode="RGB", radius_clip=3,
                    )
                else:
                    render_colors, _, _ = rasterization_fn(
                        means, quats, scales, opacities, colors,
                        viewmat[None], K[None], width, height,
                        sh_degree=sh_degree,
                        render_mode="RGB", radius_clip=3,
                    )
                self.last_render = render_colors[0, ..., :3].cpu().numpy()
                return self.last_render

        viewer_render_fn = ViewerRenderer()

        server = viser.ViserServer(port=args.port, verbose=False)
        _ = nerfview.Viewer(server=server, render_fn=viewer_render_fn, mode="rendering")

        def toggle_confidence(event):
            viewer_render_fn.show_confidence = not viewer_render_fn.show_confidence
            if viewer_render_fn.show_confidence:
                viewer_render_fn.render_data = update_conf_view(viewer_render_fn.threshold)
            cam = event.client.camera
            pos = np.array(cam.position)
            cam.position = (pos + np.random.normal(0, 1e-4, size=3))

        def update_threshold(event):
            viewer_render_fn.threshold = event.target.value
            viewer_render_fn.render_data = update_conf_view(viewer_render_fn.threshold, viewer_render_fn.show_confidence)
            cam = event.client.camera
            pos = np.array(cam.position)
            cam.position = (pos + np.random.normal(0, 1e-4, size=3))

        def capture_frame(event):
            if viewer_render_fn.last_render is not None:
                os.makedirs("captures", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                imageio.imwrite(f"captures/capture_{timestamp}.png", (np.clip(viewer_render_fn.last_render, 0.0, 1.0) * 255).astype(np.uint8))
                print(f"Saved capture to captures/capture_{timestamp}.png")

        server.gui.add_button("Toggle Confidences' heatmap view").on_click(toggle_confidence)
        server.gui.add_button("Capture Screenshot").on_click(capture_frame)
        conf_threshold_slider = server.gui.add_slider(
            label="# Confidence Threshold",
            min=0.0, max=1.0, step=0.05, initial_value=0.0,
            marks=[(0.0, "0"), (0.5, "0.5"), (0.8, "0.8"), (1.0, "1")]
        )
        conf_threshold_slider.on_update(update_threshold)

        print("Viewer running... Ctrl+C to exit.")
        time.sleep(100000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--scene_grid", type=int, default=1)
    parser.add_argument("--ckpt", type=str, nargs="+", default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", type=str, default="gsplat")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"
    cli(main, args, verbose=True)
