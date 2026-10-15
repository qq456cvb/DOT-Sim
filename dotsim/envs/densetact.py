"""DenseTact gel simulation as Gymnasium environments.

The gel is an MPM point cloud whose lowest 5% (along +y) is glued to the sensor base. An indenter (a sphere, or a
triangle mesh) is driven by velocity actions; after each step the DenseTact camera's depth and normal images are
rendered. Observations are H x W x 4 arrays: depth (0 where a ray hits nothing), then the normal.

Actions are 6-vectors: linear velocity (3) and angular velocity in axis-angle form (3), in simulation units per second.
Lengths are in units of ``scale`` millimetres (60 in the paper), so the 5 mm sphere indenter has radius 5 / 60.
"""

import gymnasium as gym
import numpy as np
import point_cloud_utils as pcu
import torch
import warp as wp
from gymnasium import spaces

from ..mpm import MeshStruct, MPMBaseStruct, MPMModelStruct, MPMStateStruct, MPMWARPDiff, SphereStruct, freeze_bottom
from ..renderer import MPMRenderer

# The DenseTact camera sits this far above the base centre (simulation units).
CAMERA_OFFSET = np.array([0., 0.02, 0.])


def load_ray_maps(phi_path, theta_path):
    """Per-pixel ray directions of the DenseTact camera, from the sensor calibration files.

    ``theta`` in the files is the elevation; the renderer uses the polar angle, ``pi / 2 - theta``.
    """
    phi = np.load(phi_path)
    theta = np.pi / 2 - np.load(theta_path)
    return phi, theta


class DenseTactBaseEnv(gym.Env):
    """Shared simulation, rendering and stepping; subclasses choose the indenter and the reward."""

    def __init__(self, scale, dx, fps, substep, E, nu, gravity, max_t,
                 particle_init_position, particle_init_velocity, ray_maps, device="cuda"):
        super().__init__()
        self.scale = scale
        self.dx = dx
        self.fps = fps
        self.substep = substep
        self.dt = 1.0 / fps
        self.substep_size = self.dt / substep
        self.E = E
        self.nu = nu
        self.gravity = gravity
        self.max_t = max_t
        self.t = 0
        self.device = device
        self.action_space = spaces.Box(low=-1., high=1., shape=(6,), dtype=np.float32)
        self.init_particles(particle_init_position, particle_init_velocity)
        self.init_renderer(*ray_maps)

    def make_primitive(self):
        raise NotImplementedError

    def init_particles(self, particle_init_position: torch.Tensor, particle_init_velocity: torch.Tensor):
        device = self.device
        n = particle_init_position.shape[0]
        grid_lims = torch.stack([torch.min(particle_init_position, 0)[0], torch.max(particle_init_position, 0)[0]])
        mpm_model = MPMModelStruct()
        mpm_model.init(n, device=device, requires_grad=False)
        mpm_model.init_other_params(grid_lims, self.dx, device=device)

        mpm_state = MPMStateStruct()
        mpm_state.init(n, device=device, requires_grad=False)
        points_volume = torch.full((n,), 1e-6, dtype=torch.float32, device=device)
        mpm_state.from_torch(
            particle_init_position,
            points_volume,
            None,
            None,
            n_grid=[mpm_model.grid_dim_x, mpm_model.grid_dim_y, mpm_model.grid_dim_z],
            device=device,
            requires_grad=False
        )

        self.base = MPMBaseStruct()
        mpm_solver = MPMWARPDiff(n, self.base)
        mpm_solver.grid_postprocess = []
        mpm_solver.collider_params = []
        mpm_solver.modify_bc = []
        freeze_bottom(particle_init_position, mpm_solver, mpm_model, fraction=0.05, axis=1)

        material_params = {
            "material": "elastic",
            "g": [0.0, self.gravity, 0.0],
            "density": 1000.,  # kg / m^3
            "grid_v_damping_scale": 1.1,  # no damping if > 1.0
            "friction_angle": 0.
        }

        self.primitive = self.make_primitive()
        mpm_solver.add_primitive(self.primitive)
        mpm_solver.set_parameters_dict(mpm_model=mpm_model, mpm_state=mpm_state, kwargs=material_params, device=device)
        mpm_solver.set_E_nu_from_torch(
            mpm_model=mpm_model,
            E=torch.full_like(particle_init_position[:, 0], self.E),
            nu=torch.full_like(particle_init_position[:, 0], self.nu),
            device=device
        )
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device=device)

        particle_density = torch.full_like(particle_init_position[:, 0], material_params["density"], requires_grad=False)
        mpm_state.reset_density(particle_density, device=device, update_mass=True, requires_grad=False)
        particle_init_F = torch.repeat_interleave(torch.eye(3, device=device).unsqueeze(0), n, dim=0)
        particle_init_C = torch.zeros_like(particle_init_F)
        mpm_state.continue_from_torch(
            particle_init_position,
            particle_init_velocity,
            particle_init_F,
            particle_init_C,
            device=device,
            requires_grad=False
        )

        mpm_solver.particle_velocity_modifiers = []
        mpm_solver.particle_velocity_modifier_params = []

        self.mpm_solver = mpm_solver
        self.mpm_model = mpm_model
        self.mpm_init_state = mpm_state

    def init_base(self):
        self.base.init(
            x=torch.zeros((3,), device=self.device).float(),
            q=torch.tensor([0., 0., 0., 1.0], device=self.device).float(),
            rot_center=torch.zeros((3,), device=self.device).float(),
        )

    def init_renderer(self, phi_mapping, theta_mapping):
        self.image_res = phi_mapping.shape[::-1]  # (W, H)
        self.renderer = MPMRenderer(
            voxel_res=(self.mpm_model.grid_dim_x, self.mpm_model.grid_dim_y, self.mpm_model.grid_dim_z),
            image_res=self.image_res, phi_mapping=phi_mapping, theta_mapping=theta_mapping, device=self.device)
        self.particle_color = wp.ones_like(self.mpm_init_state.particle_x)

    def _get_obs(self):
        self.renderer.bake(self.crt_state, self.mpm_model, self.particle_color, bake_size=2)
        _, depth, normal = self.renderer.render_densetact(self.render_offset + CAMERA_OFFSET, self.mpm_model)
        normal_np = normal.cpu().numpy()
        depth_np = depth.cpu().numpy()
        depth_np[depth_np > 10] = 0
        return np.concatenate([depth_np[..., None], normal_np], -1)

    def simulate(self, action):
        """Advance one frame (``substep`` MPM steps) with the indenter moving at ``action``."""
        prev_state = self.crt_state
        action = torch.as_tensor(np.asarray(action), device=self.device).float()
        self.primitive.set_velocity(v=action[:3], w=action[3:])
        for _ in range(self.substep):
            next_state = prev_state.partial_clone(device=self.device, requires_grad=False)
            _, _, _, next_base = self.mpm_solver.p2g2p_differentiable(
                self.mpm_model,
                prev_state,
                next_state,
                self.substep_size,
                device=self.device,
            )
            prev_state = next_state
            self.primitive = self.mpm_solver.primitives[0]
            self.base = next_base
        self.crt_state = prev_state
        self.t += self.dt
        crt_pc = self.crt_state.particle_x.numpy()
        valid = np.all(np.isfinite(crt_pc))
        return crt_pc, valid


class DenseTactEnv(DenseTactBaseEnv):
    """Gel pressed by a sphere.

    ``indentor_prop`` holds ``prim_radius``, ``prim_friction``, ``prim_softness``, ``prim_init_x``, ``prim_init_q``
    (torch tensors on ``device``) and ``render_offset`` (the base centre). The reward pulls the gel towards
    ``target_pc`` (Chamfer distance) and keeps the indenter in contact.
    """

    def __init__(self, scale, dx, fps, substep, E, nu, gravity, max_t,
                 particle_init_position, particle_init_velocity, indentor_prop, target_pc, ray_maps, device="cuda"):
        super().__init__(scale, dx, fps, substep, E, nu, gravity, max_t,
                         particle_init_position, particle_init_velocity, ray_maps, device)
        h, w = self.image_res[1], self.image_res[0]
        self.observation_space = spaces.Box(low=-1., high=1., shape=(h * w * 4,), dtype=np.float32)
        self.target_pc = target_pc
        self.indentor_prop = indentor_prop
        self.render_offset = indentor_prop["render_offset"]

    def make_primitive(self):
        return SphereStruct()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.crt_state = self.mpm_init_state
        self.primitive.init(
            friction=self.indentor_prop["prim_friction"],
            softness=self.indentor_prop["prim_softness"],
            x=self.indentor_prop["prim_init_x"],
            q=self.indentor_prop["prim_init_q"],
            r=self.indentor_prop["prim_radius"],
        )
        self.init_base()
        self.t = 0
        return self._get_obs(), {}

    @torch.no_grad()
    def step(self, action):
        crt_pc, valid = self.simulate(action)
        done = self.t >= self.max_t
        chamfer = pcu.chamfer_distance(crt_pc[::10], self.target_pc[::10])
        prim_x = self.primitive.x.numpy()[0]
        prim_dist = np.abs(np.linalg.norm(prim_x - crt_pc, axis=1).min() - self.primitive.r)
        reward = - 10. * chamfer - 1. * prim_dist - 0.01 * np.linalg.norm(action)
        return self._get_obs(), reward, chamfer < 0.01, done or (not valid), {}


class DenseTactMeshEnv(DenseTactBaseEnv):
    """Gel pressed by a triangle-mesh indenter (used to simulate the recorded indentations).

    ``other_prop`` holds ``indentor_prop`` (keyword arguments of ``MeshStruct.init_from_mesh``: ``mesh``,
    ``friction``, ``softness``, ``x``, ``q``) and ``render_offset`` (the base centre). The reward is always 0.
    With ``done_as_single``, ``step`` returns the old Gym 4-tuple.
    """

    def __init__(self, scale, dx, fps, substep, E, nu, gravity, max_t,
                 particle_init_position, particle_init_velocity, other_prop, ray_maps, done_as_single=False,
                 device="cuda"):
        super().__init__(scale, dx, fps, substep, E, nu, gravity, max_t,
                         particle_init_position, particle_init_velocity, ray_maps, device)
        self.observation_space = spaces.Box(low=-1., high=1., shape=(self.image_res[1], self.image_res[0], 4),
                                            dtype=np.float32)
        self.other_prop = other_prop
        self.render_offset = other_prop["render_offset"]
        self.done_as_single = done_as_single

    def make_primitive(self):
        return MeshStruct()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.crt_state = self.mpm_init_state
        self.primitive.init_from_mesh(**self.other_prop["indentor_prop"])
        self.init_base()
        self.t = 0
        return self._get_obs(), {}

    @torch.no_grad()
    def step(self, action):
        _, valid = self.simulate(action)
        truncated = self.t >= self.max_t or (not valid)
        terminated = False
        reward = 0.
        if self.done_as_single:
            return self._get_obs(), reward, truncated or terminated, {}
        return self._get_obs(), reward, terminated, truncated, {}
