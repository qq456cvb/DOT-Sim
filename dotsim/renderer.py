"""Ray-marched depth and normal images of the simulated gel.

The particles are splatted into a distance volume on the MPM grid (``MPMRenderer.bake``); rays are then marched through
it, either from a pinhole camera (``render``) or from the DenseTact camera, whose per-pixel ray directions come from the
sensor calibration (``render_densetact``).
"""

from typing import Any

import numpy as np
import warp as wp

from .mpm import MPMModelStruct, MPMStateStruct


@wp.kernel
def smooth(sdf: wp.array(dtype=wp.float32, ndim=3), sdf_out: wp.array(dtype=wp.float32, ndim=3), model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    sum = float(0.)
    cnt = float(0.)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if grid_x + i < 0 or grid_x + i >= model.grid_dim_x or grid_y + j < 0 or grid_y + j >= model.grid_dim_y or grid_z + k < 0 or grid_z + k >= model.grid_dim_z:
                    continue
                sum += sdf[grid_x + i, grid_y + j, grid_z + k]
                cnt += 1.
    sdf_out[grid_x, grid_y, grid_z] = sum / cnt


@wp.kernel
def build_normal(sdf: wp.array(dtype=wp.float32, ndim=3),
                 normal: wp.array(dtype=wp.vec3, ndim=3),
                 model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    n = wp.vec3(0., 0., 0.)
    if grid_x > 0 and grid_x < model.grid_dim_x - 1:
        n[0] = wp.float(sdf[grid_x + 1, grid_y, grid_z] - sdf[grid_x - 1, grid_y, grid_z]) / 2.
    if grid_y > 0 and grid_y < model.grid_dim_y - 1:
        n[1] = wp.float(sdf[grid_x, grid_y + 1, grid_z] - sdf[grid_x, grid_y - 1, grid_z]) / 2.
    if grid_z > 0 and grid_z < model.grid_dim_z - 1:
        n[2] = wp.float(sdf[grid_x, grid_y, grid_z + 1] - sdf[grid_x, grid_y, grid_z - 1]) / 2.
    normal[grid_x, grid_y, grid_z] = wp.normalize(n)


@wp.kernel
def build_vol(vol: wp.array(dtype=wp.int32, ndim=3),
              particle_color: wp.array(dtype=wp.vec3),
              bake_size: wp.int32,
              state: MPMStateStruct,
              model: MPMModelStruct):
    # Each grid cell keeps the closest particle: distance in the top bits, colour in the low 24 bits.
    particle_id = wp.tid()
    p = state.particle_x[particle_id] * model.inv_dx
    color = particle_color[particle_id]
    coord = wp.vec3i(
        wp.int(wp.floor(p[0] + 0.5)), wp.int(wp.floor(p[1] + 0.5)), wp.int(wp.floor(p[2] + 0.5))
    )

    for i in range(-bake_size, bake_size + 1):
        for j in range(-bake_size, bake_size + 1):
            for k in range(-bake_size, bake_size + 1):
                grid_x = coord[0] + i - model.grid_offset_x
                grid_y = coord[1] + j - model.grid_offset_y
                grid_z = coord[2] + k - model.grid_offset_z
                if grid_x < 0 or grid_x >= model.grid_dim_x or grid_y < 0 or grid_y >= model.grid_dim_y or grid_z < 0 or grid_z >= model.grid_dim_z:
                    continue
                d = wp.length(wp.vec3(wp.float32(grid_x + model.grid_offset_x), wp.float32(grid_y + model.grid_offset_y), wp.float32(grid_z + model.grid_offset_z)) - p)
                grid_dist = wp.min(wp.int32(wp.max(0., (127. * d / float(bake_size)))), 127)
                char = wp.lshift(wp.int32(grid_dist), wp.int32(24)) + wp.lshift(wp.int32(color[0] * 255.), wp.int32(16)) \
                    + wp.lshift(wp.int32(color[1] * 255.), wp.int32(8)) + wp.int32(color[2] * 255.)
                wp.atomic_min(vol, wp.int32(grid_x), wp.int32(grid_y), wp.int32(grid_z), wp.int32(char))


@wp.kernel
def get_sdf(vol: wp.array(dtype=wp.int32, ndim=3), sdf: wp.array(dtype=wp.float32, ndim=3), color: wp.array(dtype=wp.vec3, ndim=3)):
    grid_x, grid_y, grid_z = wp.tid()
    c = vol[grid_x, grid_y, grid_z]
    for i in range(2, -1, -1):
        color[grid_x, grid_y, grid_z][i] = wp.float(c & wp.int32(255)) / 255.
        c = wp.rshift(c, wp.int32(8))
    sdf[grid_x, grid_y, grid_z] = wp.float(c & wp.int32(127)) / 127.


@wp.func
def sample_tex(tex: wp.array(dtype=Any, ndim=3), pos: wp.vec3, model: MPMModelStruct):
    # Trilinear interpolation of a grid quantity at world position `pos`.
    grid_pos = pos * model.inv_dx
    grid_pos[0] -= float(model.grid_offset_x)
    grid_pos[1] -= float(model.grid_offset_y)
    grid_pos[2] -= float(model.grid_offset_z)
    base_pos_x = wp.clamp(wp.int(wp.floor(grid_pos[0])), 0, model.grid_dim_x - 1)
    base_pos_y = wp.clamp(wp.int(wp.floor(grid_pos[1])), 0, model.grid_dim_y - 1)
    base_pos_z = wp.clamp(wp.int(wp.floor(grid_pos[2])), 0, model.grid_dim_z - 1)
    fx = grid_pos - wp.vec3(wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z))
    fx[0] = wp.clamp(fx[0], 0., 1.)
    fx[1] = wp.clamp(fx[1], 0., 1.)
    fx[2] = wp.clamp(fx[2], 0., 1.)

    x0, y0, z0 = base_pos_x, base_pos_y, base_pos_z
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    x1 = wp.min(x1, model.grid_dim_x - 1)
    y1 = wp.min(y1, model.grid_dim_y - 1)
    z1 = wp.min(z1, model.grid_dim_z - 1)

    c00 = tex[x0, y0, z0] * (1. - fx[0]) + tex[x1, y0, z0] * fx[0]
    c01 = tex[x0, y0, z1] * (1. - fx[0]) + tex[x1, y0, z1] * fx[0]
    c10 = tex[x0, y1, z0] * (1. - fx[0]) + tex[x1, y1, z0] * fx[0]
    c11 = tex[x0, y1, z1] * (1. - fx[0]) + tex[x1, y1, z1] * fx[0]

    c0 = c00 * (1. - fx[1]) + c10 * fx[1]
    c1 = c01 * (1. - fx[1]) + c11 * fx[1]

    val = c0 * (1. - fx[2]) + c1 * fx[2]
    return val


@wp.func
def sample_sdf(sdf: wp.array(dtype=wp.float32, ndim=3), pos: wp.vec3, model: MPMModelStruct):
    # The surface is the 0.4 level set of the baked distance; outside the grid counts as empty.
    if pos[0] < model.grid_lims[0][0] or pos[1] < model.grid_lims[0][1] or pos[2] < model.grid_lims[0][2] or pos[0] > model.grid_lims[1][0] or pos[1] > model.grid_lims[1][1] or pos[2] > model.grid_lims[1][2]:
        return 1. - 0.4
    return sample_tex(sdf, pos, model) - 0.4


@wp.func
def sample_color(color: wp.array(dtype=wp.vec3, ndim=3), pos: wp.vec3, model: MPMModelStruct):
    return sample_tex(color, pos, model)


@wp.func
def sample_normal(normal: wp.array(dtype=wp.vec3, ndim=3), pos: wp.vec3, model: MPMModelStruct):
    return sample_tex(normal, pos, model)


@wp.func
def ray_aabb_intersection(grid_lims: wp.array(dtype=wp.vec3), o: wp.vec3, d: wp.vec3):
    intersect = 1

    near_int = - 1e10
    far_int = 1e10

    for i in range(3):
        if d[i] == 0:
            if o[i] < grid_lims[0][i] or o[i] > grid_lims[1][i]:
                intersect = 0
        else:
            t1 = (grid_lims[0][i] - o[i]) / d[i]
            t2 = (grid_lims[1][i] - o[i]) / d[i]
            tmin = wp.min(t1, t2)
            tmax = wp.max(t1, t2)
            near_int = wp.max(near_int, tmin)
            far_int = wp.min(far_int, tmax)

    if near_int > far_int:
        intersect = 0
    return intersect, near_int, far_int


@wp.func
def next_hit(o: wp.vec3, d: wp.vec3,
             sdf: wp.array(dtype=wp.float32, ndim=3),
             color: wp.array(dtype=wp.vec3, ndim=3),
             normal: wp.array(dtype=wp.vec3, ndim=3),
             model: MPMModelStruct, inverse: wp.int32 = 0):
    # March along the ray until the SDF changes sign (inverse=1: from inside the gel outwards), then bisect back.
    n = wp.vec3(0., 0., 0.)
    col = wp.vec3(0., 0., 0.)
    closest = float(1e10)

    intersect, tnear, tfar = ray_aabb_intersection(model.grid_lims, o, d)

    if intersect != 0:
        tnear = wp.max(tnear, 0.)
        pos = o + d * (tnear + 1e-3)
        step = wp.vec3(0., 0., 0.)

        for j in range(500):
            s = sample_sdf(sdf, pos, model)
            if (s < 0 and inverse == 0) or (s > 0. and inverse == 1):
                back_step = step
                for k in range(20):
                    back_step *= 0.5
                    sdf_back = sample_sdf(sdf, pos - back_step, model)
                    if (sdf_back < 0 and inverse == 0) or (sdf_back > 0. and inverse == 1):
                        pos -= back_step

                dist = wp.length(pos - o)
                if dist < closest:
                    closest = dist
                    if closest < 10.:  # no hit: leave the normal and colour at zero
                        n = sample_normal(normal, pos, model)
                        col = sample_color(color, pos, model)
                break
            else:
                if inverse == 0:
                    step = d * wp.max(s * 0.05, 0.01)
                else:
                    step = d * wp.max(-s * 0.05, 0.01)
                pos += step

    return closest, n, col


@wp.kernel
def render_image(color_buffer: wp.array(dtype=wp.vec3, ndim=2), depth_buffer: wp.array(dtype=wp.float32, ndim=2), normal_buffer: wp.array(dtype=wp.vec3, ndim=2),
                 K_inv: wp.mat33, cam_rot: wp.mat33, cam_trans: wp.vec3,
                 model: MPMModelStruct,
                 sdf: wp.array(dtype=wp.float32, ndim=3),
                 color: wp.array(dtype=wp.vec3, ndim=3),
                 normal: wp.array(dtype=wp.vec3, ndim=3)):
    u, v = wp.tid()
    cam_ray = wp.normalize(K_inv * wp.vec3(wp.float32(u), wp.float32(v), 1.))
    world_ray = cam_rot * cam_ray

    closest, n, col = next_hit(cam_trans, world_ray, sdf, color, normal, model, 0)

    color_buffer[v, u] += col
    depth_buffer[v, u] += closest
    normal_buffer[v, u] += n


@wp.kernel
def render_image_densetact(color_buffer: wp.array(dtype=wp.vec3, ndim=2), depth_buffer: wp.array(dtype=wp.float32, ndim=2), normal_buffer: wp.array(dtype=wp.vec3, ndim=2),
                           model: MPMModelStruct,
                           sdf: wp.array(dtype=wp.float32, ndim=3),
                           color: wp.array(dtype=wp.vec3, ndim=3),
                           normal: wp.array(dtype=wp.vec3, ndim=3),
                           origin: wp.vec3, phi_mapping: wp.array(dtype=wp.float32, ndim=2), theta_mapping: wp.array(dtype=wp.float32, ndim=2)):
    u, v = wp.tid()

    phi = phi_mapping[v, u]
    theta = theta_mapping[v, u]

    world_ray = wp.vec3(wp.sin(theta) * wp.cos(phi), wp.cos(theta), wp.sin(theta) * wp.sin(phi))

    closest, n, col = next_hit(origin, world_ray, sdf, color, normal, model, 1)

    color_buffer[v, u] += col
    depth_buffer[v, u] += closest
    normal_buffer[v, u] += n


class MPMRenderer:
    """Depth/normal renderer on the MPM grid.

    ``phi_mapping`` and ``theta_mapping`` (H x W, radians) give the direction of each pixel's ray for the DenseTact
    camera; ``image_res`` is (W, H).
    """

    def __init__(self, voxel_res, image_res, phi_mapping, theta_mapping, device=None):
        assert phi_mapping.shape == (image_res[1], image_res[0]), (phi_mapping.shape, image_res)
        assert theta_mapping.shape == (image_res[1], image_res[0]), (theta_mapping.shape, image_res)
        self.device = wp.get_device(device)
        with wp.ScopedDevice(self.device):
            self.vol = wp.full((voxel_res[0], voxel_res[1], voxel_res[2]), value=(1 << 31) - 1, dtype=wp.int32, requires_grad=False)
            self.color = wp.zeros((voxel_res[0], voxel_res[1], voxel_res[2]), dtype=wp.vec3, requires_grad=False)
            self.sdf = wp.zeros((voxel_res[0], voxel_res[1], voxel_res[2]), dtype=wp.float32, requires_grad=False)
            self.sdf2 = wp.zeros((voxel_res[0], voxel_res[1], voxel_res[2]), dtype=wp.float32, requires_grad=False)
            self.normal = wp.zeros((voxel_res[0], voxel_res[1], voxel_res[2]), dtype=wp.vec3, requires_grad=False)

            self.color_buffer = wp.zeros((image_res[1], image_res[0]), dtype=wp.vec3, requires_grad=False)
            self.depth_buffer = wp.zeros((image_res[1], image_res[0]), dtype=wp.float32, requires_grad=False)
            self.normal_buffer = wp.zeros((image_res[1], image_res[0]), dtype=wp.vec3, requires_grad=False)
            self.phi_mapping = wp.from_numpy(phi_mapping, dtype=wp.float32, shape=(image_res[1], image_res[0]))
            self.theta_mapping = wp.from_numpy(theta_mapping, dtype=wp.float32, shape=(image_res[1], image_res[0]))

    def bake(self, state: MPMStateStruct, model: MPMModelStruct, particle_color, bake_size):
        """Splat the particles into the distance volume, smooth it twice and compute its normals."""
        self.vol.fill_((1 << 31) - 1)
        self.sdf.fill_(1.)
        self.sdf2.fill_(1.)
        self.normal.fill_(0.)
        self.color.fill_(0.)
        wp.launch(
            kernel=build_vol,
            dim=state.particle_x.shape[0],
            inputs=[self.vol, particle_color, bake_size, state, model],
            device=self.device
        )
        wp.launch(
            kernel=get_sdf,
            dim=self.sdf.shape,
            inputs=[self.vol, self.sdf, self.color],
            device=self.device
        )
        wp.launch(
            kernel=smooth,
            dim=self.sdf.shape,
            inputs=[self.sdf, self.sdf2, model],
            device=self.device
        )
        wp.launch(
            kernel=smooth,
            dim=self.sdf.shape,
            inputs=[self.sdf2, self.sdf, model],
            device=self.device
        )
        wp.launch(
            kernel=build_normal,
            dim=self.sdf.shape,
            inputs=[self.sdf, self.normal, model],
            device=self.device
        )
        return self.sdf, self.color

    def render(self, intrinsic, extrinsic, model: MPMModelStruct):
        """Colour, depth and normal images (torch tensors) from a pinhole camera; ``extrinsic`` is world-to-camera."""
        self.color_buffer.fill_(0.)
        self.depth_buffer.fill_(0.)
        self.normal_buffer.fill_(0.)

        intrinsic_inv = np.linalg.inv(intrinsic)
        K_inv = wp.mat33(intrinsic_inv[0, 0], intrinsic_inv[0, 1], intrinsic_inv[0, 2],
                         intrinsic_inv[1, 0], intrinsic_inv[1, 1], intrinsic_inv[1, 2],
                         intrinsic_inv[2, 0], intrinsic_inv[2, 1], intrinsic_inv[2, 2])
        cam_pos = np.linalg.inv(extrinsic)
        cam_rot = wp.mat33(cam_pos[0, 0], cam_pos[0, 1], cam_pos[0, 2],
                           cam_pos[1, 0], cam_pos[1, 1], cam_pos[1, 2],
                           cam_pos[2, 0], cam_pos[2, 1], cam_pos[2, 2])
        cam_trans = wp.vec3(cam_pos[0, 3], cam_pos[1, 3], cam_pos[2, 3])
        wp.launch(
            kernel=render_image,
            dim=(self.color_buffer.shape[1], self.color_buffer.shape[0]),
            inputs=[self.color_buffer, self.depth_buffer, self.normal_buffer, K_inv, cam_rot, cam_trans,
                    model, self.sdf, self.color, self.normal],
            device=self.device
        )
        return wp.to_torch(self.color_buffer), wp.to_torch(self.depth_buffer), wp.to_torch(self.normal_buffer)

    def render_densetact(self, origin, model: MPMModelStruct):
        """Colour, depth and normal images (torch tensors) seen by the DenseTact camera at ``origin``."""
        self.color_buffer.fill_(0.)
        self.depth_buffer.fill_(0.)
        self.normal_buffer.fill_(0.)

        origin = wp.vec3(origin[0], origin[1], origin[2])
        wp.launch(
            kernel=render_image_densetact,
            dim=(self.color_buffer.shape[1], self.color_buffer.shape[0]),
            inputs=[self.color_buffer, self.depth_buffer, self.normal_buffer,
                    model, self.sdf, self.color, self.normal,
                    origin, self.phi_mapping, self.theta_mapping],
            device=self.device
        )
        return wp.to_torch(self.color_buffer), wp.to_torch(self.depth_buffer), wp.to_torch(self.normal_buffer)
