"""Simulate the recorded indentations (Sec. IV-B data): replay every sequence of a trajectory with its mesh indenter
and save the simulated depth and normal images next to the real tactile images.

For each sequence ``<k>`` of trajectory ``<t>`` this writes, to ``<out_root>_E<E>_nu<nu>[_dedup]/<dataset_id>/``:

    t<t>_seq<k>_gt_colors.mp4   real tactile images (640 x 480)
    t<t>_seq<k>_depths.mp4      simulated depth x 3, clipped to [0, 1]
    t<t>_seq<k>_normals.mp4     simulated normals mapped to [0, 1]

Expected data (lengths in mm):

    <data_root>/dataset_<id>/traj2ind_<id>.json      trajectory id -> indenter id
    <data_root>/dataset_<id>/t<t>_matched.csv        one row per frame: ImageName ("seq<k>_..."), Ind1_x/y/z,
                                                     Ind1_roll_deg/pitch_deg/yaw_deg (indenter pose)
    <data_root>/dataset_<id>/t<t>/t<t>_error_log.csv frames to skip (column new_name, "x<n>")
    <data_root>/dataset_<id>/t<t>/x<n>.npy, x<n>.png gel point cloud and tactile image of frame n
    <data_root>/indenters/indenter_<i>.obj           indenter meshes
    <phi_map>, <theta_map>                           DenseTact camera ray directions (H x W)
"""

import argparse
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial.transform import Rotation as R

from dotsim.envs import DenseTactMeshEnv, load_ray_maps
from dotsim.geometry import compute_angular_velocity, downsample

SIM = dict(dx=0.02, fps=24, substep=100, gravity=0., max_t=0.67, scale=60.)
IMAGE_SIZE = (640, 480)


def write_video(path, frames, fps):
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def process_seq(seq_name, traj_id, df_all, valid_names, seq_names, wd, indenter_path, E, nu, out_dir, ray_maps,
                deduplicate_gt, device):
    scale, fps = SIM["scale"], SIM["fps"]
    dt = 1.0 / fps
    mask = seq_names == seq_name
    df = df_all[mask]
    frame_names = valid_names[mask]

    # The gel at the first frame, as the simulation's rest shape.
    base_points = np.load(wd / f"{frame_names[0]}.npy") / scale
    xyz = torch.from_numpy(base_points[downsample(base_points, 0.01)]).to(device).float()

    ind_xyz = np.stack([df["Ind1_x"], df["Ind1_y"], df["Ind1_z"]], axis=1) / scale
    ind_rpy = np.stack([df["Ind1_roll_deg"], df["Ind1_pitch_deg"], df["Ind1_yaw_deg"]], axis=1)
    ind_poses = [{"rot": R.from_euler("xyz", ind_rpy[i], degrees=True).as_quat(), "tr": ind_xyz[i]}
                 for i in range(len(df))]

    indentor_mesh = trimesh.load(indenter_path)
    indentor_mesh.apply_scale(1.0 / scale)

    env = DenseTactMeshEnv(
        scale=scale, dx=SIM["dx"], fps=fps, substep=SIM["substep"], E=E, nu=nu, gravity=SIM["gravity"],
        max_t=SIM["max_t"], particle_init_position=xyz, particle_init_velocity=torch.zeros_like(xyz),
        other_prop={
            "indentor_prop": {
                "mesh": indentor_mesh,
                "friction": 0.9,
                "softness": 15.,
                "x": torch.tensor(ind_poses[0]["tr"], device=device).float(),
                "q": torch.tensor(ind_poses[0]["rot"], device=device).float(),
            },
            "render_offset": np.zeros(3),
        },
        ray_maps=ray_maps,
        device=device,
    )

    gt_img = cv2.imread(str(wd / f"{frame_names[0]}.png"))
    if gt_img is None:
        print(f"GT image {wd / f'{frame_names[0]}.png'} not found.")
        return
    gt_colors = [cv2.resize(gt_img, IMAGE_SIZE) / 255.0]

    obs, _ = env.reset()
    depth, normal = np.split(obs, [1], axis=-1)
    depths, normals = [depth], [normal]
    for i in range(len(df) - 1):
        action = np.concatenate([
            (np.array(ind_poses[i + 1]["tr"]) - np.array(ind_poses[i]["tr"])) / dt,
            compute_angular_velocity(ind_poses[i]["rot"], ind_poses[i + 1]["rot"], dt),
        ])
        obs, _, _, _, _ = env.step(action)
        depth, normal = np.split(obs, [1], axis=-1)

        gt_img = cv2.imread(str(wd / f"{frame_names[i + 1]}.png"))
        if gt_img is None:
            print(f"GT image {wd / f'{frame_names[i + 1]}.png'} not found.")
            continue
        gt_colors.append(cv2.resize(gt_img, IMAGE_SIZE) / 255.0)
        normals.append(normal)
        depths.append(depth)
    print(f"{len(gt_colors)} frames in total")

    if deduplicate_gt:
        # keep only the first of identical camera frames
        seen, keep = set(), []
        for i, color in enumerate(gt_colors):
            digest = hashlib.sha1(color.tobytes()).digest()
            if digest not in seen:
                seen.add(digest)
                keep.append(i)
        gt_colors = [gt_colors[i] for i in keep]
        normals = [normals[i] for i in keep]
        depths = [depths[i] for i in keep]
        print(f"Deduplicated gt colors, {len(gt_colors)} frames left")

    stem = f"{out_dir}/t{traj_id}_seq{seq_name}"
    write_video(f"{stem}_normals.mp4",
                [(np.clip((n + 1.) / 2., 0, 1) * 255).astype(np.uint8) for n in normals], fps / 10)
    write_video(f"{stem}_depths.mp4", [(np.clip(d * 3, 0, 1) * 255).astype(np.uint8) for d in depths], fps / 10)
    write_video(f"{stem}_gt_colors.mp4", [(np.clip(c, 0, 1) * 255).astype(np.uint8) for c in gt_colors], fps / 10)
    print(f"Finished processing t{traj_id}_seq{seq_name}")


def process_seq_unpack(args_tuple):
    return process_seq(*args_tuple)


def main():
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--traj_id", type=lambda s: [int(x) for x in s.split(",")], default=[1],
                        help="comma-separated trajectory ids, e.g. 1,3,5")
    parser.add_argument("--E", type=float, default=27575.0, help="Young's modulus from the calibration")
    parser.add_argument("--nu", type=float, default=0.303, help="Poisson's ratio from the calibration")
    parser.add_argument("--data_root", default=os.environ.get("DOTSIM_DATA_ROOT", "data"))
    parser.add_argument("--phi_map", default=None, help="default: <data_root>/phi_mask.npy")
    parser.add_argument("--theta_map", default=None, help="default: <data_root>/theta_mask.npy")
    parser.add_argument("--out_root", default="outputs/img_pred/data")
    parser.add_argument("--deduplicate_gt", action="store_true")
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_dir = data_root / f"dataset_{args.dataset_id}"
    traj2ind = json.load(open(raw_dir / f"traj2ind_{args.dataset_id}.json"))
    ray_maps = load_ray_maps(args.phi_map or data_root / "phi_mask.npy", args.theta_map or data_root / "theta_mask.npy")
    out_dir = f"{args.out_root}_E{args.E}_nu{args.nu}{'_dedup' if args.deduplicate_gt else ''}/{args.dataset_id}"
    os.makedirs(out_dir, exist_ok=True)

    for traj_id in args.traj_id:
        indenter_id = traj2ind[str(traj_id)]
        if isinstance(indenter_id, list):
            print(f"Trajectory {traj_id} has multiple indenters, skipping...")
            continue
        wd = raw_dir / f"t{traj_id}"
        df_all = pd.read_csv(raw_dir / f"t{traj_id}_matched.csv")
        err_logs = pd.read_csv(wd / f"t{traj_id}_error_log.csv")

        invalid_rows = [int(name[1:]) - 1 for name in err_logs["new_name"].values]
        valid_idx = sorted(set(range(len(df_all))) - set(invalid_rows))
        valid_names = np.array([f"x{i + 1}" for i in valid_idx])
        df_all = df_all.drop(invalid_rows)
        seq_names = np.array([int(name.split("_")[0][3:]) for name in df_all["ImageName"].values])

        indenter_path = data_root / "indenters" / f"indenter_{indenter_id}.obj"
        jobs = [(seq, traj_id, df_all, valid_names, seq_names, wd, indenter_path, args.E, args.nu, out_dir, ray_maps,
                 args.deduplicate_gt, args.device) for seq in np.unique(seq_names)]
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            list(executor.map(process_seq_unpack, jobs))


if __name__ == "__main__":
    main()
