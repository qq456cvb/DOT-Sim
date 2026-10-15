"""Smoke test of the calibration script on a synthetic recording: data loading, forward pass, gradients, update."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import DEVICE, make_dome

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate  # noqa: E402

SCALE = 60.


def write_recording(root, traj=3, seq=1, frames=4):
    """A 6 mm gel dome pressed 1 mm per frame; the "scans" are the rest shape squashed a little more each frame."""
    wd = root / f"t{traj}"
    wd.mkdir(parents=True)
    dome = make_dome(radius=6. / SCALE) * SCALE
    rng = np.random.default_rng(0)
    rows = []
    for k in range(frames):
        squashed = dome * np.array([1., 1. - 0.03 * k, 1.]) + rng.normal(0., 0.05, dome.shape)  # scan noise (mm)
        np.save(wd / f"x{k + 1}.npy", squashed)
        rows.append({"ImageName": f"seq{seq}_{k:04d}.png", "new_name": f"x{k + 1}",
                     "Ind1_x": 0., "Ind1_y": 6. - 1. * k, "Ind1_z": 0.})
    pd.DataFrame(rows).to_csv(wd / f"t{traj}_matched.csv", index=False)


def test_calibration_runs_and_updates(tmp_path):
    write_recording(tmp_path)
    ind_xyz, gt_xyzs = calibrate.load_sequence(tmp_path, traj=3, seq=1, scale=SCALE)
    assert ind_xyz.shape == (4, 3) and len(gt_xyzs) == 4

    xyz = gt_xyzs[0][calibrate.downsample(gt_xyzs[0], 0.01)]
    sim = calibrate.Calibration(xyz, indenter_center=np.array([0., 3., 0.]) / SCALE, indenter_radius=3. / SCALE,
                                E=3e4, nu=0.3, substep=20, device=DEVICE)  # >= 20 for a stable time step
    logs = []
    E, nu = calibrate.calibrate(sim, ind_xyz, gt_xyzs, E=3e4, nu=0.3, num_iters=2, e_lr=1e-1, nu_lr=1e-2,
                                log=logs.append)
    assert len(logs) == 2 and all("mean EMD" in line for line in logs)
    assert np.isfinite(E) and np.isfinite(nu) and 0.01 <= nu <= 0.5
    assert (E, nu) != (3e4, 0.3), "one gradient step should change the parameters"
