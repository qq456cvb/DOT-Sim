import warp as wp
import warp.torch
import torch
from typing import Any
from torch import Tensor
import trimesh

from .warp_utils import from_torch_safe

@wp.func
def qmul(q1: wp.vec4, q2: wp.vec4):
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return wp.vec4(x, y, z, w)

@wp.func
def qrot(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    x, y, z, w = q[0], q[1], q[2], q[3]
    uv = wp.cross(wp.vec3(x, y, z), v)
    uuv = wp.cross(wp.vec3(x, y, z), uv)
    uv *= 2.0 * w
    uuv *= 2.0
    return v + uv + uuv

@wp.func
def qinv(q: wp.vec4) -> wp.vec4:
    return wp.vec4(-q[0], -q[1], -q[2], q[3])

@wp.struct
class CapsuleStruct(object):
    friction: float
    softness: float
    x: wp.vec3
    q: wp.vec4

    v: wp.vec3
    w: wp.vec3  # angular velocity in axis angle representation

    h: float
    r: float

    def init(self, friction: float, softness: float, x: Tensor, q: Tensor, h: float, r: float):
        self.friction = friction
        self.softness = softness
        self.x = wp.vec3(x[0], x[1], x[2])
        self.q = wp.vec4(q[0], q[1], q[2], q[3])

        self.h = h
        self.r = r

    def set_volecity(self, v: Tensor, w: Tensor):
        self.v = wp.vec3(v[0], v[1], v[2])
        self.w = wp.vec3(w[0], w[1], w[2])

    def set_position(self, x: Tensor, q: Tensor):
        self.x = wp.vec3(x[0], x[1], x[2])
        self.q = wp.vec4(q[0], q[1], q[2], q[3])

    def forward_kinematics(self, dt: float):
        self.x += self.v * dt
        self.q = qmul(axis_angle2quat(self.w * dt), self.q)

# cloning torch tensor
@wp.struct
class MeshStruct(object):
    friction: float
    softness: float
    verts: wp.array(dtype=wp.vec3)
    triangles: wp.array(dtype=wp.vec3i)

    # assume all triangles share the same velocity
    x: wp.array(dtype=wp.vec3)       # Center position
    q: wp.array(dtype=wp.vec4)      # Orientation quaternion
    v: wp.array(dtype=wp.vec3)
    w: wp.array(dtype=wp.vec3)

    force: wp.array(dtype=wp.vec3)  # use array to allow inplace update
    torque: wp.array(dtype=wp.vec3)  # use array to allow inplace update

    drivable: wp.bool
    requires_grad: wp.bool

    def init_from_mesh(self, mesh: trimesh.Trimesh, friction: float, softness: float, x: Tensor, q: Tensor, drivable=False, requires_grad=False):
        self.friction = friction
        self.softness = softness
        device = wp.device_from_torch(x.device)
        self.verts = from_torch_safe(torch.tensor(mesh.vertices, device=x.device).float(), dtype=wp.vec3, requires_grad=requires_grad)
        self.triangles = from_torch_safe(torch.tensor(mesh.faces, device=x.device).int(), dtype=wp.vec3i, requires_grad=requires_grad)
        self.x = from_torch_safe(x[None].float().clone(), dtype=wp.vec3, requires_grad=requires_grad)
        self.q = from_torch_safe(q[None].float().clone(), dtype=wp.vec4, requires_grad=requires_grad)
        self.v = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.w = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.drivable = drivable
        self.requires_grad = requires_grad

        self.clear_forces(device)

    def set_velocity(self, v: Tensor, w: Tensor):
        self.v = from_torch_safe(v[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)
        self.w = from_torch_safe(w[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)

    def set_position(self, x: Tensor, q: Tensor):
        self.x = from_torch_safe(x[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)
        self.q = from_torch_safe(q[None].float().clone(), dtype=wp.vec4, requires_grad=self.requires_grad)

    def partial_clone(self, device="cuda:0", requires_grad=False):
        new_mesh = MeshStruct()
        new_mesh.init_from_mesh(mesh=None, friction=self.friction, softness=self.softness, x=torch.zeros(3, device=device), q=torch.zeros(4, device=device), drivable=self.drivable, requires_grad=requires_grad)
        wp.copy(new_mesh.v, self.v)
        wp.copy(new_mesh.w, self.w)
        wp.copy(new_mesh.force, self.force)
        wp.copy(new_mesh.torque, self.torque)
        return new_mesh

    def clear_forces(self, device=None):
        self.force = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=self.requires_grad)
        self.torque = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=self.requires_grad)


# only wp.array allow in-place update
@wp.struct
class BoxStruct(object):
    friction: float
    softness: float
    mass: float
    inertia: wp.array(dtype=wp.vec3)
    x: wp.array(dtype=wp.vec3)       # Center position of the box
    q: wp.array(dtype=wp.vec4)      # Orientation quaternion
    v: wp.array(dtype=wp.vec3)       # Linear velocity
    w: wp.array(dtype=wp.vec3)      # Angular velocity (axis-angle representation)
    dimensions: wp.array(dtype=wp.vec3)  # Half-lengths along each axis (length/2, width/2, height/2)

    force: wp.array(dtype=wp.vec3)  # use array to allow inplace update
    torque: wp.array(dtype=wp.vec3)  # use array to allow inplace update

    drivable: wp.bool
    requires_grad: wp.bool

    def init_default(self, device='cuda:0', drivable=False, requires_grad=False):
        self.friction = 0.
        self.softness = 0.
        self.x = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.q = wp.zeros((1,), dtype=wp.vec4, device=device, requires_grad=requires_grad)
        self.v = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.w = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.mass = 0.1
        self.inertia = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.dimensions = wp.zeros((1,), dtype=wp.vec3)

        self.drivable = drivable
        self.requires_grad = requires_grad
        self.clear_forces()

    def init(self, friction: float, softness: float, mass: float, x: Tensor, q: Tensor, dimensions: Tensor, drivable=False, requires_grad=False):
        self.friction = friction
        self.softness = softness
        self.requires_grad = requires_grad
        self.mass = mass
        self.dimensions = from_torch_safe(dimensions[None].float().clone(), dtype=wp.vec3, requires_grad=requires_grad)
        self.x = from_torch_safe(x[None].float().clone(), dtype=wp.vec3, requires_grad=requires_grad)
        self.q = from_torch_safe(q[None].float().clone(), dtype=wp.vec4, requires_grad=requires_grad)
        self.v = wp.zeros((1,), dtype=wp.vec3, requires_grad=requires_grad)
        self.w = wp.zeros((1,), dtype=wp.vec3, requires_grad=requires_grad)

        # calculate box intertia tensor
        full_dim = wp.vec3(2.0 * dimensions[0].item(), 2.0 * dimensions[1].item(), 2.0 * dimensions[2].item())

        # really ought to calculate this in the init function
        # Moment of inertia for a box about its center
        I_x = (1.0/12.0) * self.mass * (full_dim[1] * full_dim[1] + full_dim[2] * full_dim[2])
        I_y = (1.0/12.0) * self.mass * (full_dim[0] * full_dim[0] + full_dim[2] * full_dim[2])
        I_z = (1.0/12.0) * self.mass * (full_dim[0] * full_dim[0] + full_dim[1] * full_dim[1])
        self.inertia = wp.array((wp.vec3(I_x, I_y, I_z) * 10.0,), dtype=wp.vec3, requires_grad=requires_grad)

        self.clear_forces()

        self.drivable = drivable

    def set_velocity(self, v: Tensor, w: Tensor):
        self.v = from_torch_safe(v[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)
        self.w = from_torch_safe(w[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)

    def set_position(self, x: Tensor, q: Tensor):
        self.x = from_torch_safe(x[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)
        self.q = from_torch_safe(q[None].float().clone(), dtype=wp.vec4, requires_grad=self.requires_grad)

    def clear_forces(self, device=None):
        self.force = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=self.requires_grad)
        self.torque = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=self.requires_grad)

    def partial_clone(self, device="cuda:0", requires_grad=False):
        new_box = BoxStruct()
        new_box.init_default(device=device, drivable=self.drivable, requires_grad=requires_grad)
        new_box.friction = self.friction
        new_box.softness = self.softness
        wp.copy(new_box.force, self.force)
        wp.copy(new_box.torque, self.torque)
        return new_box

@wp.kernel
def forward_kinematics_driven(prim: Any, prim_out: Any, dt: float):
    # Kinematically driven primitive (the indenters): move with the velocity set by the caller.
    prim_out.x[0] = prim.x[0] + prim.v[0] * dt
    q = qmul(axis_angle2quat(prim.w[0] * dt), prim.q[0])
    prim_out.q[0] = wp.normalize(q)


@wp.kernel
def forward_kinematics(prim: Any, prim_out: Any, dt: float):
    # Dynamic primitive with mass and inertia (BoxStruct): integrate contact forces and gravity.
    gravity = wp.vec3(0., -9.8 * 0.1, 0.)
    prim_out.v[0] = prim.v[0] + prim.force[0] / prim.mass * dt + gravity * dt
    prim_out.x[0] = prim.x[0] + prim_out.v[0] * dt

    inertia = prim.inertia[0]

    ang_acc = wp.vec3(
        prim.torque[0][0] / inertia[0],
        prim.torque[0][1] / inertia[1],
        prim.torque[0][2] / inertia[2]
    )
    # Update angular velocity:
    prim_out.w[0] = prim.w[0] + ang_acc * dt
    q = qmul(axis_angle2quat(prim_out.w[0] * dt), prim.q[0])
    prim_out.q[0] = wp.normalize(q)


@wp.struct
class MPMBaseStruct:
    x: wp.array(dtype=wp.vec3)       # location
    rot_center: wp.vec3
    q: wp.array(dtype=wp.vec4)       # Orientation quaternion
    v: wp.array(dtype=wp.vec3)       # Linear velocity
    w: wp.array(dtype=wp.vec3)      # Angular velocity (axis-angle representation)
    requires_grad: wp.bool

    def init_default(self, device="cuda:0", requires_grad=False):
        self.x = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.q = wp.zeros((1,), dtype=wp.vec4, device=device, requires_grad=requires_grad)
        self.rot_center = wp.vec3(0., 0., 0.)
        self.v = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.w = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.requires_grad = requires_grad

    def init(self, x: Tensor, q: Tensor, rot_center: Tensor, requires_grad=False):
        self.requires_grad = requires_grad
        self.x = from_torch_safe(x[None].float().clone(), dtype=wp.vec3, requires_grad=requires_grad)
        self.q = from_torch_safe(q[None].float().clone(), dtype=wp.vec4, requires_grad=requires_grad)
        self.rot_center = wp.vec3(rot_center[0], rot_center[1], rot_center[2])
        device = wp.device_from_torch(x.device)
        self.v = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.w = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)

    def set_velocity(self, v: Tensor, w: Tensor):
        self.v = from_torch_safe(v[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)
        self.w = from_torch_safe(w[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)

    def set_position(self, x: Tensor, q: Tensor):
        self.x = from_torch_safe(x[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)
        self.q = from_torch_safe(q[None].float().clone(), dtype=wp.vec4, requires_grad=self.requires_grad)

    def partial_clone(self, device="cuda:0", requires_grad=False):
        new_mpm = MPMBaseStruct()
        new_mpm.init_default(device=device, requires_grad=requires_grad)
        wp.copy(new_mpm.v, self.v)
        wp.copy(new_mpm.w, self.w)
        new_mpm.rot_center = self.rot_center
        return new_mpm

@wp.kernel
def forward_kinematics_base(mpm: MPMBaseStruct, mpm_out: MPMBaseStruct, dt: float):
    mpm_out.x[0] = mpm.x[0] + mpm.v[0] * dt
    q = qmul(axis_angle2quat(mpm.w[0] * dt), mpm.q[0])
    mpm_out.q[0] = wp.normalize(q)


@wp.struct
class SphereStruct(object):
    friction: float
    softness: float
    x: wp.array(dtype=wp.vec3)  # Center position of the sphere
    q: wp.array(dtype=wp.vec4)  # Orientation quaternion
    v: wp.array(dtype=wp.vec3)  # Linear velocity
    w: wp.array(dtype=wp.vec3)  # Angular velocity (axis-angle representation)
    r: float  # Radius of the sphere

    force: wp.array(dtype=wp.vec3)  # Force applied to the sphere
    torque: wp.array(dtype=wp.vec3)  # Torque applied to the sphere

    drivable: wp.bool
    requires_grad: wp.bool

    def init_default(self, device='cuda:0', drivable=False, requires_grad=False):
        self.friction = 0.
        self.softness = 0.
        self.x = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.q = wp.zeros((1,), dtype=wp.vec4, device=device, requires_grad=requires_grad)
        self.v = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.w = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.r = 0.

        self.drivable = drivable
        self.requires_grad = requires_grad
        self.clear_forces()

    def init(self, friction: float, softness: float, x: Tensor, q: Tensor, r: float, drivable=False, requires_grad=False):
        self.friction = friction
        self.softness = softness
        self.requires_grad = requires_grad
        self.r = r

        device = wp.device_from_torch(x.device)
        self.x = from_torch_safe(x[None].float().clone(), dtype=wp.vec3, requires_grad=requires_grad)
        self.q = from_torch_safe(q[None].float().clone(), dtype=wp.vec4, requires_grad=requires_grad)
        self.v = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.w = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.clear_forces(device)

        self.drivable = drivable

    def set_velocity(self, v: Tensor, w: Tensor):
        self.v = from_torch_safe(v[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)
        self.w = from_torch_safe(w[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)

    def set_position(self, x: Tensor, q: Tensor):
        self.x = from_torch_safe(x[None].float().clone(), dtype=wp.vec3, requires_grad=self.requires_grad)
        self.q = from_torch_safe(q[None].float().clone(), dtype=wp.vec4, requires_grad=self.requires_grad)

    def clear_forces(self, device=None):
        self.force = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=self.requires_grad)
        self.torque = wp.zeros((1,), dtype=wp.vec3, device=device, requires_grad=self.requires_grad)

    def partial_clone(self, device="cuda:0", requires_grad=False):
        new_sphere = SphereStruct()
        new_sphere.init_default(device=device, drivable=self.drivable, requires_grad=requires_grad)
        new_sphere.friction = self.friction
        new_sphere.softness = self.softness
        wp.copy(new_sphere.force, self.force)
        wp.copy(new_sphere.torque, self.torque)
        return new_sphere


@wp.func
def axis_angle2quat(axis_angle: wp.vec3) -> wp.vec4:
    angle = wp.length(axis_angle)
    if angle < 1e-9:
        return wp.vec4(0., 0., 0., 1.)
    axis = axis_angle / angle
    half_angle = angle * 0.5
    s = wp.sin(half_angle)
    return wp.vec4(axis[0] * s, axis[1] * s, axis[2] * s, wp.cos(half_angle))


@wp.func
def sdf(prim: SphereStruct, x_local: wp.vec3) -> wp.float32:
    return wp.length(x_local) - prim.r

@wp.func
def sdf(prim: BoxStruct, x_local: wp.vec3) -> wp.float32:
    # Transform point into the box's local coordinate frame
    # Compute the distance to the box
    dx = wp.abs(x_local[0]) - prim.dimensions[0][0]
    dy = wp.abs(x_local[1]) - prim.dimensions[0][1]
    dz = wp.abs(x_local[2]) - prim.dimensions[0][2]
    return wp.max(dx, wp.max(dy, dz))

@wp.func
def point_triangle_distance(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    # Compute the point-to-triangle distance (unsigned) using barycentrics.
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.length(ap)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.length(bp)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.length(cp)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        projection = a + v * ab
        return wp.length(p - projection)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        projection = a + w * ac
        return wp.length(p - projection)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        projection = b + w * (c - b)
        return wp.length(p - projection)

    # Otherwise, the projection lies inside the triangle.
    n = wp.normalize(wp.cross(ab, ac))
    return wp.abs(wp.dot(ap, n))


@wp.func
def ray_triangle_intersect(orig: wp.vec3, dir: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.bool:
    # Möller–Trumbore ray-triangle intersection.
    epsilon = 1e-6
    edge1 = b - a
    edge2 = c - a
    h = wp.cross(dir, edge2)
    a_val = wp.dot(edge1, h)
    if wp.abs(a_val) < epsilon:
        return False  # Ray is parallel to the triangle.
    f = 1.0 / a_val
    s = orig - a
    u = f * wp.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = wp.cross(s, edge1)
    v = f * wp.dot(dir, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * wp.dot(edge2, q)
    if t > epsilon:
        return True  # There is an intersection.
    return False

# TODO: potential improvements: precompute SDF; or approx sdf when points are far
@wp.func
def sdf(prim: MeshStruct, x_local: wp.vec3) -> wp.float32:
    # Use a ray in a fixed direction (here, +X) to decide inside/outside.
    ray_dir = wp.vec3(1.0, 0.0, 0.0)
    intersections = int(0)
    num_triangles = prim.triangles.shape[0]
    min_dist = float(1e10)  # start with a large value

    # Loop over every triangle in the mesh.
    for i in range(num_triangles):
        # Get the triangle vertex indices.
        tri = prim.triangles[i]
        a = prim.verts[tri.x]
        b = prim.verts[tri.y]
        c = prim.verts[tri.z]

        # Compute unsigned distance to this triangle.
        d = point_triangle_distance(x_local, a, b, c)
        if d < min_dist:
            min_dist = d

        # Count ray-triangle intersections.
        if ray_triangle_intersect(x_local, ray_dir, a, b, c):
            intersections += 1

    # If an odd number of intersections, the point is inside.
    inside = (intersections % 2) == 1
    if inside:
        return -min_dist
    return min_dist


@wp.func
def normal(box: BoxStruct, x_local: wp.vec3) -> wp.vec3:

    # Determine the closest face and compute its normal
    abs_x_local = wp.abs(x_local)
    if abs_x_local[0] - box.dimensions[0][0] > max(abs_x_local[1] - box.dimensions[0][1], abs_x_local[2] - box.dimensions[0][2]):
        normal_local = wp.vec3(wp.sign(x_local[0]), 0.0, 0.0)
    elif abs_x_local[1] - box.dimensions[0][1] > max(abs_x_local[0] - box.dimensions[0][0], abs_x_local[2] - box.dimensions[0][2]):
        normal_local = wp.vec3(0.0, wp.sign(x_local[1]), 0.0)
    else:
        normal_local = wp.vec3(0.0, 0.0, wp.sign(x_local[2]))

    # Transform normal back to world space
    normal_world = qrot(box.q[0], normal_local)
    return normal_world

@wp.func
def normal(prim: CapsuleStruct, x_local: wp.vec3) -> wp.vec3:
    p = wp.vec3(x_local[0], x_local[1], x_local[2])
    p[1] += prim.h / 2.
    p[1] -= wp.min(wp.max(p[1], 0.), prim.h)
    return wp.normalize(p)


@wp.func
def normal(prim: SphereStruct, x_local: wp.vec3) -> wp.vec3:
    return wp.normalize(x_local)


@wp.func
def normal(prim: MeshStruct, x_local: wp.vec3) -> wp.vec3:

    # Use a fixed ray direction (+X) for the inside/outside test.
    ray_dir = wp.vec3(1.0, 0.0, 0.0)
    intersections = int(0)
    min_dist = float(1e10)
    best_normal = wp.vec3(0.0, 0.0, 0.0)
    num_triangles = prim.triangles.shape[0]

    # Loop over all triangles in the mesh.
    for i in range(num_triangles):
        tri = prim.triangles[i]
        a = prim.verts[tri.x]
        b = prim.verts[tri.y]
        c = prim.verts[tri.z]

        # Compute the unsigned distance from x_local to this triangle.
        d = point_triangle_distance(x_local, a, b, c)
        if d < min_dist:
            min_dist = d
            # Compute the triangle's outward normal (assuming proper winding).
            best_normal = wp.normalize(wp.cross(b - a, c - a))

        # Count intersections for inside/outside determination.
        if ray_triangle_intersect(x_local, ray_dir, a, b, c):
            intersections += 1

    # Determine if the point is inside (odd number of intersections).
    inside = (intersections % 2) == 1
    if inside:
        best_normal = -best_normal

    # Transform the normal back to world space.
    normal_world = qrot(prim.q[0], best_normal)
    return normal_world


@wp.kernel
def box_floor_collision(
    box: BoxStruct,
    floor_y: float,
    wall_x: float,
    dt: float
):
    """
    Force-based collision response for the floor at y = floor_y and wall at x = wall_x.
    k_penetration is the collision stiffness (larger => stiffer penalty).
    k_friction is friction scale (larger => stronger tangential damping).
    """

    k_penetration = 1e3  # collision stiffness
    k_friction = 0.4  # friction scale

    tid = wp.tid()


    # Each of the 8 corners is processed in parallel
    # tid in [0..7] => we extract bits to get corner signs
    corner_local = wp.vec3(
        (2.0 * wp.float32((tid >> 0) & 1) - 1.0) * box.dimensions[0][0],
        (2.0 * wp.float32((tid >> 1) & 1) - 1.0) * box.dimensions[0][1],
        (2.0 * wp.float32((tid >> 2) & 1) - 1.0) * box.dimensions[0][2],
    )


    # corner_local = wp.vec3(0., 0., 0.)
    # Corner in world space
    corner_world = box.x[0] + qrot(box.q[0], corner_local)

    # Corner’s velocity in world space: v_com + ω × r
    # (where r = corner_world - box center)
    corner_r = corner_world - box.x[0]
    corner_vel = box.v[0] + wp.cross(box.w[0], corner_r)

    # --------------------
    # 1) FLOOR COLLISION
    # --------------------
    penetration_floor = floor_y - corner_world[1]

    if (penetration_floor > 0.0):
        # Normal is +Y
        n_floor = wp.vec3(0.0, 1.0, 0.0)

        # Penalty force ∝ penetration depth
        f_floor = k_penetration * penetration_floor * n_floor

        # Friction
        # Decompose corner_vel into normal/tangent parts:
        normal_v = wp.dot(corner_vel, n_floor) * n_floor
        tang_v   = corner_vel - normal_v
        # Simple Coulomb friction model:
        f_friction = wp.vec3(0.0, 0.0, 0.0)
        tang_len = wp.length(tang_v)
        if (tang_len > 1e-6):
            # negative direction of tang_v
            f_friction = -k_friction * tang_v

        total_force = f_floor + f_friction

        c_damp = 50.0
        f_damp = -c_damp * corner_vel

        total_force += f_damp

        # print total force

        # Torque about the box center
        torque_floor = wp.cross(corner_r, total_force)

        # Atomically accumulate force and torque
        wp.atomic_add(box.force, 0, total_force)
        wp.atomic_add(box.torque, 0, torque_floor)

    # --------------------
    # 2) WALL COLLISION
    # --------------------
    # We assume the wall is at x = wall_x, normal = +X
    penetration_wall = wall_x - corner_world[0]
    if (penetration_wall > 0.0):
        # Normal is +X
        n_wall = wp.vec3(1.0, 0.0, 0.0)

        # Penalty force ∝ penetration depth
        f_wall = k_penetration * penetration_wall * n_wall

        # Friction
        normal_v = wp.dot(corner_vel, n_wall) * n_wall
        tang_v   = corner_vel - normal_v
        f_friction = wp.vec3(0.0, 0.0, 0.0)
        tang_len = wp.length(tang_v)
        if (tang_len > 1e-6):
            f_friction = -k_friction * tang_v

        total_force = f_wall + f_friction
        torque_wall = wp.cross(corner_r, total_force)

        wp.atomic_add(box.force, 0, total_force)
        wp.atomic_add(box.torque, 0, torque_wall)


    # TODO -- for some reason the forces from this function are cleared before integration. Likely a bug
    # in the function which calls this code.

    # box.v[0] += box.force[0] / box.mass * dt
    # box.x[0] += box.v[0] * dt

    # box.w[0] += box.torque[0] /box.mass * dt
    # box.q[0] = qmul(axis_angle2quat(box.w[0] * dt), box.q[0])
    # box.q[0] = wp.normalize(box.q[0])  # Ensure quaternion normalization


@wp.func
def collide(prim: Any, grid_mass: float, x: wp.vec3, v_out: wp.vec3, dt: wp.float32):
    # # Apply force and torque to the rigid object
    # torque = wp.cross(x - rigid.x, force)
    x_canonical = qrot(qinv(prim.q[0]), x - prim.x[0])
    dist = sdf(prim, x_canonical)

    # Only apply collision if close enough to the surface
    if dist > 0.1:  # Clearly outside - no collision
        return v_out

    # Compute influence only for contact/near-contact cases
    # For positive distances (outside), use minimal influence
    influence = wp.float32(0.0)
    if dist <= 0.0:
        influence = wp.min(wp.exp(-prim.softness * wp.max(dist, 0.0)), 1.0)
    else:
        influence = wp.max(0.0, wp.exp(-prim.softness * dist * 10.0))

    if influence < 0.01:  # Reduced threshold for better separation
        return v_out

    #           x[0], x[1], x[2], x_canonical[0], x_canonical[1], x_canonical[2], box.x[0][0], box.x[0][1], box.x[0][2])

    n = qrot(prim.q[0], normal(prim, x_canonical))

    # Compute collider velocity at contact point using rigid body kinematics
    # v_point = v_center + ω × (point - center)
    r = x - prim.x[0]  # vector from primitive center to contact point
    collider_v = prim.v[0] + wp.cross(prim.w[0], r)

    rel_v = v_out - collider_v
    normal_v_norm = wp.dot(rel_v, n)

    if normal_v_norm < 0: # there is pressure/contact
        tangent_v = rel_v - normal_v_norm * n

        if prim.drivable:
            # compute normal force for the primitive (reaction to pressure)
            prim_force = normal_v_norm * n  # Force on primitive opposite to pressure

        tangent_v_norm = wp.length(tangent_v)
        if tangent_v_norm > 1e-9:
            if prim.drivable:
                # Apply friction force opposite to tangent velocity
                friction_magnitude = prim.friction * wp.abs(normal_v_norm)
                prim_force += friction_magnitude * tangent_v / tangent_v_norm

            # Compute friction-modified tangent velocity
            tangent_v_friction = tangent_v / tangent_v_norm * wp.max(0., tangent_v_norm + prim.friction * normal_v_norm)

            # Only blend velocities when actually in contact (penetrating)
            if dist <= 0.0:
                v_out = collider_v + rel_v * (1.0 - influence) + tangent_v_friction * influence
            else:
                # Apply friction but don't blend with collider velocity when not in contact
                v_out = v_out - normal_v_norm * n + tangent_v_friction - tangent_v
        else:
            # No tangential motion - just handle normal collision
            if dist <= 0.0:
                v_out = collider_v + rel_v * (1.0 - influence)
            else:
                # Just remove the penetrating component
                v_out = v_out - normal_v_norm * n

        if prim.drivable:
            prim_force_grid = prim_force * influence * grid_mass
            prim_torque = wp.cross(x - prim.x[0], prim_force_grid) * 100.

            # caveat: need to use array indexing to update struct fields in-place
            wp.atomic_add(prim.force, 0, prim_force_grid)
            wp.atomic_add(prim.torque, 0, prim_torque)

    else: # particles are separating (normal_v_norm >= 0)
        # Only apply forces if still penetrating the surface
        if dist <= 0.0:
            # Apply separation force to push particles out of penetration
            separation_force = -dist * n * 10.0  # Push out of penetration
            impulse = separation_force * dt / grid_mass
            v_out = v_out + impulse

            if prim.drivable:
                # Apply reaction force to primitive
                prim_force_grid = -separation_force * grid_mass
                prim_torque = wp.cross(x - prim.x[0], prim_force_grid) * 100.
                wp.atomic_add(prim.force, 0, prim_force_grid)
                wp.atomic_add(prim.torque, 0, prim_torque)
        # Otherwise, let particles move freely without interference

    return v_out
