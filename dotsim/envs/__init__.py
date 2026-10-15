from gymnasium.envs.registration import register

from .densetact import DenseTactBaseEnv, DenseTactEnv, DenseTactMeshEnv, load_ray_maps

register(id="dotsim/DenseTact-v0", entry_point="dotsim.envs.densetact:DenseTactEnv")
register(id="dotsim/DenseTact-mesh-v0", entry_point="dotsim.envs.densetact:DenseTactMeshEnv")

__all__ = ["DenseTactBaseEnv", "DenseTactEnv", "DenseTactMeshEnv", "load_ray_maps"]
