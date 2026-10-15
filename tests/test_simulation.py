"""Smoke tests of the simulation and renderer on a small synthetic gel (run on the CPU when there is no GPU)."""

import numpy as np
import torch
import trimesh

from conftest import DEVICE, DOME_RADIUS
from dotsim.envs import DenseTactEnv, DenseTactMeshEnv

SIM = dict(scale=60., dx=0.02, fps=24, substep=20, E=27575.0, nu=0.303, gravity=0., max_t=1.0)
PRESS = np.array([0., -0.6, 0., 0., 0., 0.])  # downwards, simulation units per second
TOP = DOME_RADIUS + 1 / 60  # just above the top of the dome


def depth_change(env, frames=3):
    obs0, _ = env.reset()
    for _ in range(frames):
        out = env.step(PRESS)
    obs = out[0]
    assert obs.shape == obs0.shape and np.isfinite(obs).all()
    hit = (obs0[..., 0] > 0) & (obs[..., 0] > 0)
    assert hit.mean() > 0.5, "most rays should hit the gel"
    return obs0, obs, out


def test_sphere_indenter_presses_gel(dome, ray_maps):
    radius = 5. / 60
    env = DenseTactEnv(
        **SIM, particle_init_position=dome, particle_init_velocity=torch.zeros_like(dome),
        indentor_prop={
            "prim_radius": radius, "prim_friction": 0.9, "prim_softness": 15.,
            "prim_init_x": torch.tensor([0., TOP + radius, 0.], device=DEVICE),
            "prim_init_q": torch.tensor([0., 0., 0., 1.], device=DEVICE),
            "render_offset": np.zeros(3),
        },
        target_pc=dome.cpu().numpy(), ray_maps=ray_maps, device=DEVICE)
    obs0, obs, (_, reward, _, truncated, _) = depth_change(env)
    assert np.isfinite(reward) and not truncated

    # the indenter follows the commanded velocity
    expected = TOP + radius + PRESS[1] * 3 / SIM["fps"]
    assert abs(env.primitive.x.numpy()[0][1] - expected) < 1e-4

    # the gel moved towards the camera under the indenter (the top rows look straight up)
    top = slice(0, 3)
    assert (obs[top, :, 0] < obs0[top, :, 0] - 1e-3).mean() > 0.5


def test_mesh_indenter_presses_gel(dome, ray_maps):
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=5. / 60)
    env = DenseTactMeshEnv(
        **SIM, particle_init_position=dome, particle_init_velocity=torch.zeros_like(dome),
        other_prop={
            "indentor_prop": {
                "mesh": mesh, "friction": 0.9, "softness": 15.,
                "x": torch.tensor([0., TOP + 5. / 60, 0.], device=DEVICE),
                "q": torch.tensor([0., 0., 0., 1.], device=DEVICE),
            },
            "render_offset": np.zeros(3),
        },
        ray_maps=ray_maps, device=DEVICE)
    obs0, obs, (_, reward, terminated, truncated, _) = depth_change(env)
    assert reward == 0. and not terminated and not truncated
    top = slice(0, 3)
    assert (obs[top, :, 0] < obs0[top, :, 0] - 1e-3).mean() > 0.5
