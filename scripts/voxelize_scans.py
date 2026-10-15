"""Turn the gel scans (STL surfaces) of each trajectory into filled point clouds for the calibration.

Every ``<data_root>/t<traj>/<name>.stl`` is voxelised and saved as ``<data_root>/t<traj>/<name>.npy`` (N x 3, mm).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pyvista as pv
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_root", default=os.environ.get("DOTSIM_DATA_ROOT", "data"))
    parser.add_argument("--trajs", type=lambda s: [int(x) for x in s.split(",")], default=[4, 5, 6, 7],
                        help="comma-separated trajectory ids")
    parser.add_argument("--density", type=float, default=0.2, help="voxel size (mm)")
    args = parser.parse_args()

    for traj in args.trajs:
        wd = Path(args.data_root) / f"t{traj}"
        for stl in tqdm(sorted(wd.glob("*.stl")), desc=f"t{traj}"):
            voxels = pv.voxelize(pv.read(stl), density=args.density)
            np.save(wd / f"{stl.stem}.npy", voxels.cell_centers().points)


if __name__ == "__main__":
    main()
