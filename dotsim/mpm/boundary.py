"""Grid boundary conditions for the gel.

Adapted from ``apply_grid_bc_w_freeze_pts`` in PhysDreamer (projects/inference/local_utils.py), changed to work with the
offset, non-cubic grids of :class:`MPMModelStruct`.
"""

import torch


def apply_grid_bc_w_freeze_pts(freeze_pts, mpm_solver, mpm_model):
    """Fix the grid velocity to zero in every grid cell that contains one of ``freeze_pts``.

    Returns the boolean mask of frozen grid cells.
    """
    device = freeze_pts.device
    offset = torch.tensor([mpm_model.grid_offset_x, mpm_model.grid_offset_y, mpm_model.grid_offset_z], device=device)
    dims = torch.tensor([mpm_model.grid_dim_x, mpm_model.grid_dim_y, mpm_model.grid_dim_z], device=device)
    cells = (freeze_pts * mpm_model.inv_dx + 0.5).long() - offset
    cells = torch.minimum(torch.clamp(cells, min=0), dims - 1)

    freeze_grid_mask = torch.zeros(tuple(dims.tolist()), dtype=torch.bool, device=device)
    freeze_grid_mask[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    mpm_solver.enforce_grid_velocity_by_mask(freeze_grid_mask.type(torch.int32))
    return freeze_grid_mask


def freeze_bottom(particles, mpm_solver, mpm_model, fraction=0.05, axis=1):
    """Glue the gel to its base: freeze the grid cells of the lowest ``fraction`` of particles along ``axis``."""
    coord = particles[:, axis]
    lo, hi = torch.min(coord), torch.max(coord)
    freeze_pts = particles[coord < lo + (hi - lo) * fraction]
    return apply_grid_bc_w_freeze_pts(freeze_pts, mpm_solver, mpm_model)
