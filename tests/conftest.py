import numpy as np
import pytest
import torch
import warp as wp

from dotsim.geometry import downsample

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session", autouse=True)
def warp_init():
    wp.init()


DOME_RADIUS = 0.2


def make_dome(radius=DOME_RADIUS, spacing=0.01, seed=0):
    """A solid hemispherical gel (flat side down, +y up), like the voxelised scans."""
    r = np.arange(-radius, radius + 1e-9, spacing / 2)
    xyz = np.stack(np.meshgrid(r, np.arange(0, radius + 1e-9, spacing / 2), r, indexing="ij"), -1).reshape(-1, 3)
    xyz = xyz[np.linalg.norm(xyz, axis=1) <= radius]
    return xyz[downsample(xyz, spacing, seed=seed)]


def make_ray_maps(width=24, height=16):
    """Rays fanning out over the upper hemisphere, in the format of the sensor calibration files."""
    phi = np.tile(np.linspace(0, 2 * np.pi, width, endpoint=False, dtype=np.float32), (height, 1))
    polar = np.tile(np.linspace(0.05, 1.3, height, dtype=np.float32)[:, None], (1, width))
    return phi, polar


@pytest.fixture
def dome():
    return torch.from_numpy(make_dome()).float().to(DEVICE)


@pytest.fixture
def ray_maps():
    return make_ray_maps()
