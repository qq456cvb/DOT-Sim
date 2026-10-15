"""Differentiable MLS-MPM in NVIDIA Warp, with rigid indenters.

Adapted from PhysDreamer's ``warp_mpm`` (itself based on warp-mpm); see README.md in this folder for the provenance and
the changes made for DOT-Sim.
"""

from .boundary import apply_grid_bc_w_freeze_pts, freeze_bottom
from .mpm_data_structure import MPMModelStruct, MPMStateStruct
from .mpm_solver_diff import MPMWARPDiff
from .primitive import BoxStruct, CapsuleStruct, MeshStruct, MPMBaseStruct, SphereStruct

__all__ = [
    "MPMModelStruct",
    "MPMStateStruct",
    "MPMWARPDiff",
    "MPMBaseStruct",
    "SphereStruct",
    "MeshStruct",
    "BoxStruct",
    "CapsuleStruct",
    "apply_grid_bc_w_freeze_pts",
    "freeze_bottom",
]
