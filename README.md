# DOT-Sim

**Differentiable Optical Tactile Simulation with Precise Real-to-Sim Physical Calibration**
Yang You, Won Kyung Do, Aiden Swann, Rika Antonova, Monroe Kennedy, Leonidas Guibas · ICRA 2026

[Paper](https://arxiv.org/abs/2604.27367) · [Project page](https://qq456cvb.github.io/projects/dotsim)

DOT-Sim simulates soft optical tactile sensors in two stages. The gel is an elastic material simulated with the
Material Point Method (MPM) in [NVIDIA Warp](https://github.com/NVIDIA/warp), and its material parameters are
calibrated to real recordings in minutes by differentiating through the simulation. A network then predicts the tactile
image as a residual on top of the real idle image, from the depth and normals of the simulated gel.

This repository contains the physical calibration (Sec. III-A, IV-A) and the residual optical simulation (Sec. III-B,
IV-B) for the DenseTact sensor.

## Installation

Python 3.10+ and an NVIDIA GPU. Install PyTorch for your CUDA version, then the package:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the versions used for the paper; `pip install -e .` alone installs compatible versions.

## Pipeline

All lengths in the data are in millimetres; the simulation works in units of 60 mm (`--scale`).

**1. Voxelise the gel scans** into filled point clouds (`t<traj>/x<n>.stl` → `x<n>.npy`):

```bash
python scripts/voxelize_scans.py --data_root data --trajs 4,5,6,7
```

**2. Calibrate the gel** (Young's modulus E and Poisson's ratio ν) on one recorded sequence. The gel is pressed by a
5 mm sphere following the recorded indenter trajectory, and E and ν follow the gradient of the Earth Mover's distance
between the simulated and scanned gel:

```bash
python scripts/calibrate.py --data_root data --traj 3 --seq 1
# -> outputs/calibration/e0.1_nu0.01/t3_s1_E_nu.txt
```

**3. Simulate the recordings** with the calibrated parameters and each sequence's mesh indenter, rendering the depth
and normals seen by the DenseTact camera:

```bash
python scripts/simulate_dataset.py --dataset_id <id> --traj_id 1,2,3,4,5,6 --E 27575 --nu 0.303
# -> outputs/img_pred/data_E27575.0_nu0.303/<id>/t<t>_seq<k>_{gt_colors,depths,normals}.mp4
```

**4. Pack training data** into WebDataset shards, given split files that list `*_gt_colors.mp4` videos
(`val_<name>.txt` and `train_<name>.txt`):

```bash
python scripts/build_webdataset.py --data_root outputs/img_pred/data_E27575.0_nu0.303/<id> \
    --split_dir <split folder> --split_fn val_t1_t3.txt
```

**5. Train the residual optical model** (DeepLabV3-ResNet50; configuration in `configs/optical.yaml`):

```bash
python scripts/train_optical.py data_root=outputs/img_pred/data_E27575.0_nu0.303/<id>/val_t1_t3_webdataset_pil
```

**6. Evaluate** (L2, PSNR, and L2 over the pixels that change):

```bash
python scripts/eval_optical.py --checkpoint <run>/checkpoints/last.ckpt --split_fn <split folder>/val_t1_t3.txt
```

## Data format

```
data/
├── phi_mask.npy, theta_mask.npy     DenseTact camera: per-pixel ray azimuth and elevation (H x W, radians)
├── indenters/indenter_<i>.obj       indenter meshes (mm)
├── t<traj>/                         calibration recordings
│   ├── t<traj>_matched.csv          one row per frame: ImageName ("seq<k>_..."), new_name ("x<n>"), Ind1_x/y/z
│   └── x<n>.stl, x<n>.npy           gel scan of frame n and its voxelised point cloud
└── dataset_<id>/                    recordings for the optical model
    ├── traj2ind_<id>.json           trajectory id -> indenter id
    ├── t<t>_matched.csv             as above, plus Ind1_roll_deg/pitch_deg/yaw_deg
    └── t<t>/
        ├── t<t>_error_log.csv       frames to skip (column new_name)
        └── x<n>.npy, x<n>.png       gel point cloud and tactile image of frame n
```

`--data_root` (or the `DOTSIM_DATA_ROOT` environment variable) points at `data/`; `simulate_dataset.py` also takes
`--phi_map` and `--theta_map` for the camera files.

## Code

```
dotsim/
├── mpm/            differentiable MPM in Warp with rigid indenters (adapted from PhysDreamer, see mpm/README.md)
├── envs/           the DenseTact gel as Gymnasium environments (sphere or mesh indenter; depth + normal images)
├── renderer.py     ray-marched depth and normals for the DenseTact camera
├── optical/        residual optical model and its data loading
└── geometry.py     point-cloud and rotation helpers
scripts/            the pipeline above
configs/            training configuration
tests/              smoke tests on a small synthetic gel
```

The tests run on the GPU if there is one and on the CPU otherwise (a few minutes):

```bash
pip install pytest
pytest
```

## Citation

```bibtex
@inproceedings{you2026dotsim,
  title={DOT-Sim: Differentiable Optical Tactile Simulation with Precise Real-to-Sim Physical Calibration},
  author={You, Yang and Do, Won Kyung and Swann, Aiden and Antonova, Rika and Kennedy, Monroe and Guibas, Leonidas},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026},
  note={arXiv:2604.27367}
}
```

## License and acknowledgements

MIT (see `LICENSE`). The MPM solver in `dotsim/mpm/` is adapted from
[PhysDreamer](https://github.com/a1600012888/PhysDreamer), which builds on
[warp-mpm](https://github.com/zeshunzong/warp-mpm) and [NCLaw](https://github.com/PingchuanMa/NCLaw); these projects
do not state a license, and their attribution is kept in `dotsim/mpm/README.md`.
