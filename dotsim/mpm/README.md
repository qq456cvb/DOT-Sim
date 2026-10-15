# Differentiable MPM

This folder is adapted from the `physdreamer/warp_mpm` module of
[PhysDreamer](https://github.com/a1600012888/PhysDreamer) (Zhang et al., 2024), which is in turn based on
[warp-mpm](https://github.com/zeshunzong/warp-mpm) (Zong et al.) and [NCLaw](https://github.com/PingchuanMa/NCLaw)
(Ma et al.). We thank the authors of these projects. Neither upstream repository states a license; the files keep
their original structure so the origin of each part stays traceable.

Changes for DOT-Sim:

- `primitive.py` (new): rigid indenters as Warp structs (sphere, triangle mesh with a signed distance computed from
  the triangles, box), their contact response on the MPM grid (`collide`), and their motion each substep (spheres
  and meshes follow the velocity they are given; boxes are integrated with their contact forces).
- Grids fitted to the gel: non-cubic dimensions with an offset around the particles' bounding box
  (`MPMModelStruct.init_other_params`), used throughout the particle-to-grid and grid-to-particle kernels.
- Gradients through contact: every substep works on fresh copies of the state (`partial_clone_particle`), so Warp's
  tape can differentiate through the collision and boundary-condition passes; `copy_grad_from` chains gradients
  between frames.
- An `elastic` material (neo-Hookean, `material = 7`) and a mask-based boundary condition that glues the gel to its
  base (`boundary.py`, from PhysDreamer's `apply_grid_bc_w_freeze_pts`).
