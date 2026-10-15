"""Physical calibration (Sec. III-A / IV-A): fit Young's modulus E and Poisson's ratio nu of the gel to one recording.

A recorded sequence gives the indenter trajectory and, for every frame, the scanned gel shape. The gel is simulated
with a 5 mm sphere indenter following the trajectory, and E and nu are updated by gradient descent on the Sinkhorn
(Earth Mover's) distance between the simulated and scanned point clouds; gradients come from Warp's automatic
differentiation through the MPM steps. The fitted values are written to ``<out_dir>/t<traj>_s<seq>_E_nu.txt``.

Expected data (lengths in mm), from ``scripts/voxelize_scans.py``:

    <data_root>/t<traj>/t<traj>_matched.csv   one row per frame: ImageName ("seq<k>_..."), new_name ("x<n>"),
                                              Ind1_x, Ind1_y, Ind1_z (indenter position)
    <data_root>/t<traj>/x<n>.npy              scanned gel of frame n, N x 3 points
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import warp as wp
from geomloss import SamplesLoss
from tqdm import tqdm

from dotsim.geometry import downsample
from dotsim.mpm import MPMBaseStruct, MPMModelStruct, MPMStateStruct, MPMWARPDiff, SphereStruct, freeze_bottom

# Gradients are passed from one frame back to the previous one, except across every third frame, where the chain is
# cut to keep the backward pass stable.
TRUNCATE_EVERY = 3


def load_sequence(data_root, traj, seq, scale):
    """Indenter positions (F x 3) and scanned gel point clouds (F arrays of N x 3) of one sequence, in sim units."""
    wd = Path(data_root) / f"t{traj}"
    df = pd.read_csv(wd / f"t{traj}_matched.csv")
    seq_names = np.array([int(name.split("_")[0][3:]) for name in df["ImageName"].values])
    df = df[seq_names == seq]
    frame_names = [int(name[1:]) for name in df["new_name"].values]
    ind_xyz = np.stack([df["Ind1_x"], df["Ind1_y"], df["Ind1_z"]], axis=1) / scale
    gt_xyzs = [np.load(wd / f"x{fn}.npy") / scale for fn in frame_names]
    return ind_xyz, gt_xyzs


class Calibration:
    def __init__(self, xyz, indenter_center, indenter_radius, E, nu, dx=0.02, fps=24, substep=100, device="cuda:0"):
        self.device = device
        self.dt = 1.0 / fps
        self.substep = substep
        self.substep_size = self.dt / substep
        self.indenter_center = indenter_center
        self.indenter_radius = indenter_radius

        grid_lims = torch.from_numpy(np.stack([np.min(xyz, axis=0), np.max(xyz, axis=0)])).to(device).float()
        x0 = torch.from_numpy(xyz).to(device).float()
        n = x0.shape[0]
        self.x0 = x0

        mpm_model = MPMModelStruct()
        mpm_model.init(n, device=device, requires_grad=True)
        mpm_model.init_other_params(grid_lims=grid_lims, dx=dx, device=device)

        mpm_state = MPMStateStruct()
        mpm_state.init(n, device=device, requires_grad=True)
        mpm_state.from_torch(
            x0,
            torch.full((n,), 1e-6, dtype=torch.float32, device=device),
            None,
            None,
            n_grid=[mpm_model.grid_dim_x, mpm_model.grid_dim_y, mpm_model.grid_dim_z],
            device=device,
            requires_grad=True
        )

        base = MPMBaseStruct()
        base.init(
            x=torch.zeros((3,), device=device).float(),
            q=torch.tensor([0., 0., 0., 1.0], device=device).float(),
            rot_center=torch.zeros((3,), device=device).float(),
        )
        solver = MPMWARPDiff(n, base)
        solver.grid_postprocess = []
        solver.collider_params = []
        solver.modify_bc = []
        freeze_bottom(x0, solver, mpm_model, fraction=0.05, axis=1)

        material_params = {
            "material": "elastic",
            "g": [0.0, 0.0, 0.0],
            "density": 1000.,  # kg / m^3
            "grid_v_damping_scale": 1.1,  # no damping if > 1.0
            "friction_angle": 0.
        }
        self.primitive = SphereStruct()
        self.mpm_model, self.mpm_state, self.solver = mpm_model, mpm_state, solver
        self.reset_indenter(np.zeros(3))
        solver.add_primitive(self.primitive)
        solver.set_parameters_dict(mpm_model=mpm_model, mpm_state=mpm_state, kwargs=material_params, device=device)
        self.set_material(E, nu)

        particle_density = torch.full_like(x0[:, 0], material_params["density"], requires_grad=True)
        mpm_state.reset_density(particle_density, device=device, update_mass=True, requires_grad=True)
        particle_init_F = torch.repeat_interleave(torch.eye(3, device=device).unsqueeze(0), n, dim=0)
        particle_init_C = torch.zeros_like(particle_init_F)
        mpm_state.continue_from_torch(
            x0,
            torch.zeros_like(x0),
            particle_init_F,
            particle_init_C,
            device=device,
            requires_grad=True
        )
        solver.particle_velocity_modifiers = []
        solver.particle_velocity_modifier_params = []

    def reset_indenter(self, position):
        self.primitive.init(friction=0.9, softness=15.,
                            x=torch.tensor(self.indenter_center + position, device=self.device).float(),
                            q=torch.tensor([0., 0., 0., 1.0], device=self.device).float(), r=self.indenter_radius)

    def set_material(self, E, nu):
        self.solver.set_E_nu_from_torch(
            mpm_model=self.mpm_model,
            E=torch.full_like(self.x0[:, 0], E),
            nu=torch.full_like(self.x0[:, 0], nu),
            device=self.device
        )
        self.solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device=self.device)

    def drive(self, velocity):
        self.primitive.set_velocity(v=torch.tensor(velocity, device=self.device).float(),
                                    w=torch.zeros(3, device=self.device).float())

    def advance(self, state, requires_grad=True):
        """One frame (``substep`` MPM steps) from ``state``; returns the list of states, ``state`` first."""
        states = [state]
        for _ in range(self.substep):
            next_state = states[-1].partial_clone(device=self.device, requires_grad=requires_grad)
            self.solver.p2g2p_differentiable(self.mpm_model, states[-1], next_state, self.substep_size,
                                             device=self.device)
            states.append(next_state)
        self.primitive = self.solver.primitives[0]
        return states


def calibrate(sim, ind_xyz, gt_xyzs, E, nu, num_iters, e_lr, nu_lr, log=print):
    """Gradient descent on log(E) and nu; returns the values after ``num_iters - 1`` updates."""
    device = sim.device
    loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=0.001)
    num_frames = len(ind_xyz) - 1
    logE = np.log(E)

    for it in range(num_iters):
        sim.reset_indenter(ind_xyz[0])
        sim.mpm_model.clear_grad()
        sim.mpm_state.clear_grad()
        sim.set_material(np.exp(logE), nu)

        # Forward: simulate the whole sequence, keeping the state at the start of every frame.
        prev_state = sim.mpm_state
        cache_states = [prev_state]
        cache_prim_states = [(sim.primitive.x.numpy()[0], sim.primitive.q.numpy()[0])]
        losses = []
        for i in tqdm(range(num_frames), desc=f"iter {it} forward", leave=False):
            sim.drive((ind_xyz[i + 1] - ind_xyz[i]) / sim.dt)
            prev_state = sim.advance(prev_state, requires_grad=True)[-1]
            pos = sim.solver.get_particle_x(prev_state, device=device)
            gt_xyz = torch.from_numpy(np.array(gt_xyzs[i + 1])).float().to(device)
            losses.append(loss_fn(torch.from_numpy(pos).float().to(device), gt_xyz[::20].contiguous()).item())
            cache_states.append(prev_state)
            cache_prim_states.append((sim.primitive.x.numpy()[0], sim.primitive.q.numpy()[0]))
        log(f"iter {it}: E = {np.exp(logE):.1f}, nu = {nu:.4f}, mean EMD = {np.mean(losses):.6f}")

        if it == num_iters - 1:
            break

        # Backward: re-simulate each frame on a tape, from the last frame to the first, chaining the gradients.
        tape = None
        for i in tqdm(reversed(range(num_frames)), total=num_frames, desc=f"iter {it} backward", leave=False):
            x, q = cache_prim_states[i]
            sim.primitive.set_position(x=torch.tensor(x, device=device), q=torch.tensor(q, device=device))
            sim.drive((ind_xyz[i + 1] - ind_xyz[i]) / sim.dt)

            tape = wp.Tape()
            with tape:
                states = sim.advance(cache_states[i], requires_grad=True)

            pos = wp.to_torch(states[-1].particle_x).clone().contiguous().detach()
            pos.requires_grad = True
            gt_xyz = torch.from_numpy(np.array(gt_xyzs[i + 1])).float().to(device)
            loss_fn(pos, gt_xyz[::20].contiguous()).backward()

            if i < num_frames - 1 and i % TRUNCATE_EVERY != 0:
                states[-1].copy_grad_from(cache_states[i + 1])
                states[-1].particle_x.grad = wp.clone(wp.from_torch(
                    (pos.grad + wp.to_torch(states[-1].particle_x.grad)).contiguous(), dtype=wp.vec3))
            else:
                states[-1].particle_x.grad = wp.clone(wp.from_torch(pos.grad, dtype=wp.vec3))
            tape.backward(grads={states[-1].particle_x: wp.clone(states[-1].particle_x.grad)})

        # Chain rule from (mu, lambda) to (E, nu).
        grad_mu = tape.gradients[sim.mpm_model.mu].numpy()
        grad_lam = tape.gradients[sim.mpm_model.lam].numpy()
        tape = wp.Tape()
        with tape:
            sim.solver.prepare_mu_lam(sim.mpm_model, sim.mpm_state, device=device)
        tape.backward(grads={sim.mpm_model.mu: grad_mu, sim.mpm_model.lam: grad_lam})
        grad_E = tape.gradients[sim.mpm_model.E].numpy().sum()
        grad_nu = tape.gradients[sim.mpm_model.nu].numpy().sum()

        logE -= e_lr * grad_E * np.exp(logE)
        nu -= nu_lr * grad_nu
        nu = np.clip(nu, 0.01, 0.5)

    return np.exp(logE), nu


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_root", default=os.environ.get("DOTSIM_DATA_ROOT", "data"))
    parser.add_argument("--traj", type=int, default=3)
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--scale", type=float, default=60., help="millimetres per simulation unit")
    parser.add_argument("--indenter_radius", type=float, default=5., help="sphere indenter radius (mm)")
    parser.add_argument("--indenter_offset", type=float, default=5.,
                        help="height of the sphere centre above the tracked indenter position (mm)")
    parser.add_argument("--init_E", type=float, default=3e4)
    parser.add_argument("--init_nu", type=float, default=0.3)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--e_lr", type=float, default=1e-1)
    parser.add_argument("--nu_lr", type=float, default=1e-2)
    parser.add_argument("--substep", type=int, default=100)
    parser.add_argument("--out_dir", default=None, help="default: outputs/calibration/e<e_lr>_nu<nu_lr>")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or f"outputs/calibration/e{args.e_lr}_nu{args.nu_lr}")
    out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    ind_xyz, gt_xyzs = load_sequence(args.data_root, args.traj, args.seq, args.scale)
    xyz = gt_xyzs[0][downsample(gt_xyzs[0], 0.01)]
    print(f"{len(ind_xyz)} frames, {len(xyz)} particles")

    sim = Calibration(xyz, indenter_center=np.array([0., args.indenter_offset, 0.]) / args.scale,
                      indenter_radius=args.indenter_radius / args.scale, E=args.init_E, nu=args.init_nu,
                      substep=args.substep, device=args.device)
    E, nu = calibrate(sim, ind_xyz, gt_xyzs, args.init_E, args.init_nu, args.num_iters, args.e_lr, args.nu_lr)

    out = out_dir / f"t{args.traj}_s{args.seq}_E_nu.txt"
    out.write_text(f"{E}\n{nu}\n")
    print(f"E = {E:.1f}, nu = {nu:.4f} -> {out}")


if __name__ == "__main__":
    main()
