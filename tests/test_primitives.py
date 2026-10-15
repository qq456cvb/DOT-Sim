"""The triangle-mesh signed distance used for mesh indenters, against analytic shapes."""

import numpy as np
import torch
import trimesh
import warp as wp

from conftest import DEVICE
from dotsim.mpm import MeshStruct
from dotsim.mpm.primitive import sdf


@wp.kernel
def compute_sdf_kernel(prim: MeshStruct, points: wp.array(dtype=wp.vec3), sdf_out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    sdf_out[tid] = sdf(prim, points[tid])


def mesh_sdf(mesh, points):
    prim = MeshStruct()
    prim.init_from_mesh(mesh, friction=0.5, softness=0., x=torch.zeros(3, device=DEVICE),
                        q=torch.tensor([0., 0., 0., 1.], device=DEVICE))
    points_wp = wp.array(points.astype(np.float32), dtype=wp.vec3, device=DEVICE)
    out = wp.empty(len(points), dtype=wp.float32, device=DEVICE)
    wp.launch(compute_sdf_kernel, dim=len(points), inputs=[prim, points_wp, out], device=DEVICE)
    return out.numpy()


def test_box_sdf():
    points = np.random.default_rng(0).uniform(-1.5, 1.5, size=(100, 3))
    q = np.abs(points) - 1.0
    expected = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    got = mesh_sdf(trimesh.creation.box(extents=[2.0, 2.0, 2.0]), points)
    assert np.abs(got - expected).max() < 0.05


def test_sphere_sdf():
    points = np.random.default_rng(1).uniform(-1.5, 1.5, size=(100, 3))
    expected = np.linalg.norm(points, axis=1) - 1.0
    got = mesh_sdf(trimesh.creation.icosphere(subdivisions=3, radius=1.0), points)
    assert np.abs(got - expected).max() < 0.1  # the icosphere is slightly inside the true sphere
