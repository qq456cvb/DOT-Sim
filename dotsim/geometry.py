"""Point-cloud and rotation helpers shared by the simulation scripts.

Quaternions are stored as (x, y, z, w), as in SciPy.
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def downsample(pc, res, seed=0):
    """Indices of one random point per voxel of size ``res`` (Open3D voxel grid).

    Seeds NumPy's global generator, as the experiments in the paper did, so the same points are picked.
    """
    np.random.seed(seed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    _, _, idxs = pcd.voxel_down_sample_and_trace(res, pcd.get_min_bound(), pcd.get_max_bound())
    return np.array([np.random.choice(np.array(idx)) for idx in idxs])


def quat_tr2mat(q, tr):
    """4x4 transform from a quaternion and a translation."""
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(q).as_matrix()
    mat[:3, 3] = tr
    return mat


def qmul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w])


def qinv(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def quat2axis_angle(q):
    q = np.asarray(q, dtype=np.float64)
    norm_q = np.linalg.norm(q)
    if norm_q == 0:
        raise ValueError("Quaternion norm is zero")
    x, y, z, w = q / norm_q
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(1 - w * w)
    if s < 1e-9:
        return np.zeros(3)
    return np.array([x, y, z]) / s * angle


def compute_angular_velocity(q1, q2, dt):
    """Angular velocity (axis-angle per second) that rotates ``q1`` into ``q2`` in ``dt``."""
    return quat2axis_angle(qmul(q2, qinv(q1))) / dt
