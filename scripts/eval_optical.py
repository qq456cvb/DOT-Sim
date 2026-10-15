"""Evaluate the residual optical model (Sec. IV-B): image L2, PSNR and L2 over the pixels that change.

Each sequence in the split file is predicted frame by frame from its simulated depth and normal videos, starting
from the real first frame, and compared with the real tactile images.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from dotsim.optical import DeepLabModel, read_mp4


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint from train_optical.py")
    parser.add_argument("--config", default=None, help="config of the checkpoint (default: use_diff=true)")
    parser.add_argument("--split_fn", required=True, help="text file listing *_gt_colors.mp4 videos")
    parser.add_argument("--rescale", action="store_true",
                        help="convert the videos back to physical units (depth / 3, normals to [-1, 1]) before "
                             "feeding them to the model; by default they are fed as stored (/255), which is how "
                             "build_webdataset.py prepares the training data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config) if args.config else OmegaConf.create({"use_diff": True, "use_gt": False})
    model = DeepLabModel.load_from_checkpoint(args.checkpoint, cfg=cfg, map_location=args.device).eval().to(args.device)
    fns = [line.strip() for line in Path(args.split_fn).read_text().splitlines() if line.strip()]

    val_l2, val_psnr, val_sig_l2 = [], [], []
    with torch.no_grad():
        for fn in tqdm(fns):
            gt_colors = read_mp4(fn) / 255.0
            suffix = "_gt" if cfg.use_gt else ""
            rendered_depths = read_mp4(fn.replace("_gt_colors", f"{suffix}_depths")) / 255.0
            rendered_normals = read_mp4(fn.replace("_gt_colors", f"{suffix}_normals")) / 255.0

            if (not cfg.use_gt) and args.rescale:
                rendered_depths /= 3.0
            if args.rescale:
                rendered_normals = rendered_normals * 2.0 - 1.0

            gt_colors = torch.from_numpy(np.moveaxis(gt_colors, -1, 1)).to(args.device).float()
            rendered_depths = torch.from_numpy(np.moveaxis(rendered_depths[:, ::-1], -1, 1).copy()).to(args.device).float()
            rendered_normals = torch.from_numpy(np.moveaxis(rendered_normals[:, ::-1], -1, 1).copy()).to(args.device).float()

            pred_colors = [gt_colors[0]]
            for i in range(1, len(gt_colors)):
                if cfg.use_diff:
                    input_depth = rendered_depths[i] - rendered_depths[0]
                    input_normal = rendered_normals[i] - rendered_normals[0]
                else:
                    input_depth = rendered_depths[i]
                    input_normal = rendered_normals[i]
                pred = model(torch.cat([input_depth, input_normal], dim=0)[None])[0]
                pred_colors.append(gt_colors[0] + pred if cfg.use_diff else pred)

            pred_colors = torch.stack(pred_colors)
            l2 = torch.mean((pred_colors - gt_colors) ** 2, dim=(1, 2, 3))
            diff_mask = ((gt_colors - gt_colors[0]).abs().sum(dim=1, keepdim=True) > 0.01).float()
            sig_l2 = ((pred_colors - gt_colors) ** 2 * diff_mask).sum((1, 2, 3)) / (diff_mask.sum((1, 2, 3)).clamp(min=1e-8) * 3)
            val_l2.append(l2)
            val_psnr.append(10 * torch.log10(1 / l2.clamp(min=1e-8)))
            val_sig_l2.append(sig_l2)

    print("Val L2 norm:", torch.cat(val_l2).sqrt().mean().item())
    print("Val PSNR:", torch.cat(val_psnr).mean().item())
    print("Val significant L2 norm:", torch.cat(val_sig_l2).sqrt().mean().item())


if __name__ == "__main__":
    main()
