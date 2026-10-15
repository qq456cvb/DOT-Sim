import warp as wp
from typing import Any

from .mpm_data_structure import *
from .primitive import SphereStruct, collide, BoxStruct, qrot, MeshStruct, MPMBaseStruct

# compute stress from F
@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)

@wp.func
def kirchoff_stress_elastic(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    B = F * wp.transpose(F)
    J = wp.determinant(F)
    scale = lam * wp.log(J) - mu
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    stress = mu * B + scale * id
    # if wp.isnan(stress[0][0]):
    #     wp.printf('F: %f %f %f %f %f %f %f %f %f\n', F[0][0], F[0][1], F[0][2], F[1][0], F[1][1], F[1][2], F[2][0], F[2][1], F[2][2])
    return stress

@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    """
    B = F * wp.transpose(F)
    dev(B) = B - (1/3) * tr(B) * I

    For a compressible Rivlin neo-Hookean materia, the cauchy stress is given by:
    mu * J^(-2/3) * dev(B) + lam * J (J - 1) * I
    see: https://en.wikipedia.org/wiki/Neo-Hookean_solid
    """

    # compute kirchoff stress for FCR model (remember tau = P F^T)
    b = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3(
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
    )
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * wp.vec3(
        1.0, 1.0, 1.0
    )

    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_StVK(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (
        1.0 / sig[0]
    )
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (
        1.0 / sig[1]
    )
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (
        1.0 / sig[2]
    )
    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)


@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


@wp.func
def von_mises_return_mapping_with_damage(
    F_trial: wp.mat33, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        if model.yield_stress[p] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        model.yield_stress[p] = model.yield_stress[p] - model.softening * wp.length(
            (delta_gamma / epsilon_hat_norm) * epsilon_hat
        )
        if model.yield_stress[p] <= 0:
            model.mu[p] = 0.0
            model.lam[p] = 0.0
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


# for toothpaste
@wp.func
def viscoplasticity_return_mapping_with_StVK(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, dt: float
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    b_trial = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - wp.vec3(
        trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
    )
    s_trial = 2.0 * model.mu[p] * epsilon_hat
    s_trial_norm = wp.length(s_trial)
    y = s_trial_norm - wp.sqrt(2.0 / 3.0) * model.yield_stress[p]
    if y > 0:
        mu_hat = model.mu[p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
        s_new_norm = s_trial_norm - y / (
            1.0 + model.plastic_viscosity / (2.0 * mu_hat * dt)
        )
        s_new = (s_new_norm / s_trial_norm) * s_trial
        epsilon_new = 1.0 / (2.0 * model.mu[p]) * s_new + wp.vec3(
            trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
        )
        sig_elastic = wp.mat33(
            wp.exp(epsilon_new[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        return F_elastic
    else:
        return F_trial


@wp.func
def sand_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    epsilon = wp.vec3(
        wp.log(wp.max(wp.abs(sig[0]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[1]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[2]), 1e-14)),
    )
    sigma_out = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    epsilon_hat_norm = wp.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * model.lam[p] + 2.0 * model.mu[p])
        / (2.0 * model.mu[p])
        * tr
        * model.alpha
    )

    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))

        F_elastic = U * wp.diag(s_new) * wp.transpose(V)
    return F_elastic


@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    model.mu[p] = model.E[p] / (2.0 * (1.0 + model.nu[p]))
    model.lam[p] = (
        model.E[p] * model.nu[p] / ((1.0 + model.nu[p]) * (1.0 - 2.0 * model.nu[p]))
    )


@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx


@wp.func
def update_cov(state: MPMStateStruct, p: int, grad_v: wp.mat33, dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    state.particle_cov[p * 6] = cov_np1[0, 0]
    state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.func
def update_cov_differentiable(
    state: MPMStateStruct,
    next_state: MPMStateStruct,
    p: int,
    grad_v: wp.mat33,
    dt: float,
):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    next_state.particle_cov[p * 6] = cov_np1[0, 0]
    next_state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    next_state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    next_state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    next_state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    next_state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    # output:               grid_v_in, grid_m
    p = wp.tid()
    # if state.particle_v[p][0] != 0.0:
    #     wp.printf("particle_v %f %f %f\n", state.particle_v[p][0], state.particle_v[p][1], state.particle_v[p][2])
    if state.particle_selection[p] == 0:
        stress = state.particle_stress[p]
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(wp.floor(grid_pos[0] - 0.5))
        base_pos_y = wp.int(wp.floor(grid_pos[1] - 0.5))
        base_pos_z = wp.int(wp.floor(grid_pos[2] - 0.5))
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        # dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        affine = -dt * state.particle_vol[p] * 4.0 * model.inv_dx * model.inv_dx * stress + state.particle_mass[p] * state.particle_C[p]
        # if wp.isnan(affine[0][0]):
        #     wp.printf('particle vol (1000x): %f\n', state.particle_vol[p] * 1000.)
        #     wp.printf('particle mass: %f\n', state.particle_mass[p])
        #     wp.printf('stress: %f %f %f %f %f %f %f %f %f\n', stress[0][0], stress[0][1], stress[0][2], stress[1][0], stress[1][1], stress[1][2], stress[2][0], stress[2][1], stress[2][2])
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    dpos = (
                        wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    ) * model.dx
                    ix = base_pos_x + i - model.grid_offset_x
                    iy = base_pos_y + j - model.grid_offset_y
                    iz = base_pos_z + k - model.grid_offset_z
                    if (ix <= 0) or (iy <= 0) or (iz <= 0) or (ix >= model.grid_dim_x) or (
                        iy >= model.grid_dim_y
                    ) or (iz >= model.grid_dim_z):
                        wp.printf('p2g ix iy iz %d %d %d\n', ix, iy, iz)

                    # if ix == 55 and iz == 55:
                    #     wp.printf('particle mass %f\n', state.particle_mass[p])
                    #     wp.printf('particle vol %f\n', state.particle_vol[p])
                    #     wp.printf('dpos %f %f %f\n', dpos[0], dpos[1], dpos[2])
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    # dweight = compute_dweight(model, w, dw, i, j, k)

                    # C = state.particle_C[p]
                    # # if model.rpic = 0, standard apic
                    # C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (
                    #     C - wp.transpose(C)
                    # )

                    # # C = (1.0 - model.rpic_damping) * state.particle_C[
                    # #     p
                    # # ] + model.rpic_damping / 2.0 * (
                    # #     state.particle_C[p] - wp.transpose(state.particle_C[p])
                    # # )

                    # if model.rpic_damping < -0.001:
                    #     # standard pic
                    #     C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    # elastic_force = -state.particle_vol[p] * stress * dweight
                    # # if elastic_force > 0.0:
                    # #     wp.printf("elastic_force %f", elastic_force)
                    # v_in_add = (
                    #     weight
                    #     * state.particle_mass[p]
                    #     * (state.particle_v[p] + C * dpos)
                    #     + dt * elastic_force
                    # )
                    # if p == 0:
                    #     wp.printf('%f %f %f %f %f\n', state.particle_mass[p], state.particle_vol[p], stress, state.particle_v[p], elastic_force)
                    v_in_add = weight * (state.particle_mass[p] * state.particle_v[p] + affine * dpos)
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                    )


@wp.kernel
def mpm_to_rigid_collision(state: MPMStateStruct, out_state: MPMStateStruct, prim: Any, base: MPMBaseStruct,
                           dx: float, grid_offset_x: int, grid_offset_y: int, grid_offset_z: int, dt: float):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-8 * dx * dx * dx:
        grid_idx = wp.vec3i(grid_x + grid_offset_x, grid_y + grid_offset_y, grid_z + grid_offset_z)

        x = wp.vec3(wp.float32(grid_idx[0]), wp.float32(grid_idx[1]), wp.float32(grid_idx[2])) * dx
        x = base.x[0] + qrot(base.q[0], x - base.rot_center) + base.rot_center

        rad_v = wp.cross(base.w[0], x - base.rot_center)
        v = state.grid_v_out[grid_x, grid_y, grid_z] + base.v[0] + rad_v

        v_out = collide(prim, state.grid_m[grid_x, grid_y, grid_z], x, v, dt)  # assume all particles have the same mass

        # Update the MPM grid velocity to reflect the interaction
        out_state.grid_v_out[grid_x, grid_y, grid_z] = v_out - base.v[0] - rad_v
        # out_state.grid_v_out[grid_x, grid_y, grid_z] = state.grid_v_out[grid_x, grid_y, grid_z]
    else:
        out_state.grid_v_out[grid_x, grid_y, grid_z] = state.grid_v_out[grid_x, grid_y, grid_z]

@wp.overload
def mpm_to_rigid_collision(state: MPMStateStruct, out_state: MPMStateStruct, prim: SphereStruct, base: MPMBaseStruct,
                           dx: float, grid_offset_x: int, grid_offset_y: int, grid_offset_z: int, dt: float):
    ...

@wp.overload
def mpm_to_rigid_collision(state: MPMStateStruct, out_state: MPMStateStruct, prim: BoxStruct, base: MPMBaseStruct,
                           dx: float, grid_offset_x: int, grid_offset_y: int, grid_offset_z: int, dt: float):
    ...

@wp.overload
def mpm_to_rigid_collision(state: MPMStateStruct, out_state: MPMStateStruct, prim: MeshStruct, base: MPMBaseStruct,
                           dx: float, grid_offset_x: int, grid_offset_y: int, grid_offset_z: int, dt: float):
    ...

@wp.kernel
def get_particle_x_w_base(state: MPMStateStruct, base: MPMBaseStruct, out: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    # if p == 0:
    #     wp.printf('x %f %f %f\n', state.particle_x[p][0], state.particle_x[p][1], state.particle_x[p][2])
    #     wp.printf('base rot_center %f %f %f\n', base.rot_center[0], base.rot_center[1], base.rot_center[2])
    #     wp.printf('base q %f %f %f %f\n', base.q[0], base.q[1], base.q[2], base.q[3])
    #     rotated = qrot(base.q, state.particle_x[p] - base.rot_center) + base.rot_center
    #     wp.printf('rotated %f %f %f\n', rotated[0], rotated[1], rotated[2])
    out[p] = base.x[0] + qrot(base.q[0], state.particle_x[p] - base.rot_center) + base.rot_center
    # out[p] = state.particle_x[p] + base.x


# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-8 * model.dx * model.dx * model.dx:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * (
            1.0 / state.grid_m[grid_x, grid_y, grid_z]
        )
        if wp.isnan(v_out[0]):
            v_out = wp.vec3(0.0, 0.0, 0.0)
        # add gravity
        v_out = v_out + dt * model.gravitational_accelaration
        # if grid_x == 55 and grid_z == 55:
        #     wp.printf('%f %f %f %f\n', v_out[0], v_out[1], v_out[2], dt)
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out


@wp.kernel
def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(wp.floor(grid_pos[0] - 0.5))
        base_pos_y = wp.int(wp.floor(grid_pos[1] - 0.5))
        base_pos_z = wp.int(wp.floor(grid_pos[2] - 0.5))
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i - model.grid_offset_x
                    iy = base_pos_y + j - model.grid_offset_y
                    iz = base_pos_z + k - model.grid_offset_z
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = new_v + grid_v * weight
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    # dweight = compute_dweight(model, w, dw, i, j, k)
                    # new_F = new_F + wp.outer(grid_v, dweight)
        state.particle_v[p] = new_v
        # state.particle_x[p] = state.particle_x[p] + dt * new_v
        # state.particle_x[p] = state.particle_x[p] + dt * state.particle_v[p]

        # wp.atomic_add(state.particle_x, p, dt * state.particle_v[p]) # old one is this..
        wp.atomic_add(state.particle_x, p, dt * new_v)  # debug
        # new_x = state.particle_x[p] + dt * state.particle_v[p]
        # state.particle_x[p] = new_x

        state.particle_C[p] = new_C

        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_tmp = (I33 + new_C * dt) * state.particle_F[p]
        state.particle_F_trial[p] = F_tmp
        # debug for jelly
        # wp.atomic_add(state.particle_F_trial, p, new_F * dt * state.particle_F[p])

        # if model.update_cov_with_F:
        #     update_cov(state, p, new_F, dt)


@wp.kernel
def g2p_differentiable(
    state: MPMStateStruct, next_state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    """
    Compute:
        next_state.particle_v, next_state.particle_x, next_state.particle_C, next_state.particle_F_trial
    """
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(wp.floor(grid_pos[0] - 0.5))
        base_pos_y = wp.int(wp.floor(grid_pos[1] - 0.5))
        base_pos_z = wp.int(wp.floor(grid_pos[2] - 0.5))
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        # dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        # new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_C = wp.mat33(new_v, new_v, new_v)

        # new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i - model.grid_offset_x
                    iy = base_pos_y + j - model.grid_offset_y
                    iz = base_pos_z + k - model.grid_offset_z
                    if (ix <= 0) or (iy <= 0) or (iz <= 0) or (ix >= model.grid_dim_x) or (
                        iy >= model.grid_dim_y
                    ) or (iz >= model.grid_dim_z):
                        wp.printf('g2p ix iy iz %d %d %d\n', ix, iy, iz)
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = (
                        new_v + grid_v * weight
                    )  # TODO, check gradient from static loop
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    # dweight = compute_dweight(model, w, dw, i, j, k)
                    # new_F = new_F + wp.outer(grid_v, dweight)

        # if p == 0:
        next_state.particle_v[p] = new_v

        # add clip here:
        new_x = state.particle_x[p] + dt * new_v

        # if p == 0:
        #     wp.printf('%f %f %f\n', new_x[0] * 2. - 1., new_x[1] * 2. - 1., new_x[2] * 2. - 1.)

        # dx = 1.0 / model.inv_dx
        # a_min = dx * 2.0
        # a_max = model.grid_lim - dx * 2.0

        a_mins = model.grid_lims[0]
        a_maxs = model.grid_lims[1]

        new_x_clamped = wp.vec3(
            wp.clamp(new_x[0], a_mins[0], a_maxs[0]),
            wp.clamp(new_x[1], a_mins[1], a_maxs[1]),
            wp.clamp(new_x[2], a_mins[2], a_maxs[2]),
        )
        next_state.particle_x[p] = new_x_clamped

        # next_state.particle_x[p] = new_x

        next_state.particle_C[p] = new_C

        I33_1 = wp.vec3(1.0, 0.0, 0.0)
        I33_2 = wp.vec3(0.0, 1.0, 0.0)
        I33_3 = wp.vec3(0.0, 0.0, 1.0)
        I33 = wp.mat33(I33_1, I33_2, I33_3)
        F_tmp = (I33 + new_C * dt) * state.particle_F[p]
        next_state.particle_F_trial[p] = F_tmp

        # if 0:
        #     update_cov_differentiable(state, next_state, p, new_F, dt)


@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    """
    state.particle_F_trial => state.particle_F   # return mapping
    state.particle_F => state.particle_stress    # stress-strain

    TODO: check the gradient of SVD!  is wp.svd3 differentiable? I guess so
    """
    p = wp.tid()
    if state.particle_selection[p] == 0:
        # apply return mapping
        if model.material == 1:  # metal
            state.particle_F[p] = von_mises_return_mapping(
                state.particle_F_trial[p], model, p
            )
        elif model.material == 2:  # sand
            state.particle_F[p] = sand_return_mapping(
                state.particle_F_trial[p], state, model, p
            )
        elif model.material == 3:  # visplas, with StVk+VM, no thickening
            state.particle_F[p] = viscoplasticity_return_mapping_with_StVK(
                state.particle_F_trial[p], model, p, dt
            )
        elif model.material == 5:
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p
            )
        else:  # elastic, jelly, or neo-hookean
            state.particle_F[p] = state.particle_F_trial[p]

        # also compute stress here
        J = wp.determinant(state.particle_F[p])
        U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sig = wp.vec3(0.0)
        stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        wp.svd3(state.particle_F[p], U, sig, V)
        if model.material == 0 or model.material == 5:
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )
        if model.material == 1:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 2:
            stress = kirchoff_stress_drucker_prager(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 3:
            # temporarily use stvk, subject to change
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )

        if model.material == 6:
            stress = kirchoff_stress_neoHookean(
                state.particle_F[p], U, V, J, sig, model.mu[p], model.lam[p]
            )

        if model.material == 7:
            stress = kirchoff_stress_elastic(
                state.particle_F[p], U, V, J, sig, model.mu[p], model.lam[p]
            )
        # stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry
        # state.particle_stress[p] = (stress + wp.transpose(stress)) / 2.0
        # if p == 100:
        #     wp.printf('stress %f %f %f %f %f %f %f %f %f\n', stress[0][0], stress[0][1], stress[0][2], stress[1][0], stress[1][1], stress[1][2], stress[2][0], stress[2][1], stress[2][2])
        #     wp.printf('particle_F %f %f %f %f %f %f %f %f %f\n', state.particle_F[p][0][0], state.particle_F[p][0][1], state.particle_F[p][0][2], state.particle_F[p][1][0], state.particle_F[p][1][1], state.particle_F[p][1][2], state.particle_F[p][2][0], state.particle_F[p][2][1], state.particle_F[p][2][2])
        state.particle_stress[p] = stress


@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: float):
    grid_x, grid_y, grid_z = wp.tid()
    # state.grid_v_out[grid_x, grid_y, grid_z] = (
    #     state.grid_v_out[grid_x, grid_y, grid_z] * scale
    # )
    wp.atomic_sub(
        state.grid_v_out,
        grid_x,
        grid_y,
        grid_z,
        (1.0 - scale) * state.grid_v_out[grid_x, grid_y, grid_z],
    )


@wp.kernel
def apply_additional_params(
    state: MPMStateStruct,
    model: MPMModelStruct,
    params_modifier: MaterialParamsModifier,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if (
        pos[0] > params_modifier.point[0] - params_modifier.size[0]
        and pos[0] < params_modifier.point[0] + params_modifier.size[0]
        and pos[1] > params_modifier.point[1] - params_modifier.size[1]
        and pos[1] < params_modifier.point[1] + params_modifier.size[1]
        and pos[2] > params_modifier.point[2] - params_modifier.size[2]
        and pos[2] < params_modifier.point[2] + params_modifier.size[2]
    ):
        model.E[p] = params_modifier.E
        model.nu[p] = params_modifier.nu
        state.particle_density[p] = params_modifier.density


@wp.kernel
def selection_add_impulse_on_particles(
    state: MPMStateStruct, impulse_modifier: Impulse_modifier
):
    p = wp.tid()
    offset = state.particle_x[p] - impulse_modifier.point
    if (
        wp.abs(offset[0]) < impulse_modifier.size[0]
        and wp.abs(offset[1]) < impulse_modifier.size[1]
        and wp.abs(offset[2]) < impulse_modifier.size[2]
    ):
        impulse_modifier.mask[p] = 1
    else:
        impulse_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_translation(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_cylinder(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point

    vertical_distance = wp.abs(wp.dot(offset, velocity_modifier.normal))

    horizontal_distance = wp.length(
        offset - wp.dot(offset, velocity_modifier.normal) * velocity_modifier.normal
    )
    if (
        vertical_distance < velocity_modifier.half_height_and_radius[0]
        and horizontal_distance < velocity_modifier.half_height_and_radius[1]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0


@wp.kernel
def compute_position_l2_loss(
    mpm_state: MPMStateStruct,
    gt_pos: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    pos = mpm_state.particle_x[tid]
    pos_gt = gt_pos[tid]

    # l1_diff = wp.abs(pos - pos_gt)
    l2 = wp.length(pos - pos_gt)

    wp.atomic_add(loss, 0, l2)


@wp.kernel
def aggregate_grad(x: wp.array(dtype=float), grad: wp.array(dtype=float)):
    tid = wp.tid()

    # gradient descent step
    wp.atomic_add(x, 0, grad[tid])


@wp.kernel
def set_F_C_p2g(
    state: MPMStateStruct, model: MPMModelStruct, target_pos: wp.array(dtype=wp.vec3)
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(wp.floor(grid_pos[0] - 0.5))
        base_pos_y = wp.int(wp.floor(grid_pos[1] - 0.5))
        base_pos_z = wp.int(wp.floor(grid_pos[2] - 0.5))
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        # p2g for displacement
        particle_disp = target_pos[p] - state.particle_x[p]
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    v_in_add = weight * state.particle_mass[p] * particle_disp
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                    )


@wp.kernel
def set_F_C_g2p(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(wp.floor(grid_pos[0] - 0.5))
        base_pos_y = wp.int(wp.floor(grid_pos[1] - 0.5))
        base_pos_z = wp.int(wp.floor(grid_pos[2] - 0.5))
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # g2p for C and F
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        # C should still be zero..
        # state.particle_C[p] = new_C
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_tmp = I33 + new_F
        state.particle_F_trial[p] = F_tmp

        if model.update_cov_with_F:
            update_cov(state, p, new_F, 1.0)


@wp.kernel
def compute_posloss_with_grad(
    mpm_state: MPMStateStruct,
    gt_pos: wp.array(dtype=wp.vec3),
    grad: wp.array(dtype=wp.vec3),
    dt: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    pos = mpm_state.particle_x[tid]
    pos_gt = gt_pos[tid]

    # l1_diff = wp.abs(pos - pos_gt)
    # l2 = wp.length(pos - (pos_gt - grad[tid] * dt))
    diff = pos - (pos_gt - grad[tid] * dt)
    l2 = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, l2)


@wp.kernel
def compute_veloloss_with_grad(
    mpm_state: MPMStateStruct,
    gt_pos: wp.array(dtype=wp.vec3),
    grad: wp.array(dtype=wp.vec3),
    dt: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    pos = mpm_state.particle_v[tid]
    pos_gt = gt_pos[tid]

    # l1_diff = wp.abs(pos - pos_gt)
    # l2 = wp.length(pos - (pos_gt - grad[tid] * dt))

    diff = pos - (pos_gt - grad[tid] * dt)
    l2 = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, l2)


@wp.kernel
def compute_Floss_with_grad(
    mpm_state: MPMStateStruct,
    gt_mat: wp.array(dtype=wp.mat33),
    grad: wp.array(dtype=wp.mat33),
    dt: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    mat_ = mpm_state.particle_F_trial[tid]
    mat_gt = gt_mat[tid]

    mat_gt = mat_gt - grad[tid] * dt
    # l1_diff = wp.abs(pos - pos_gt)
    mat_diff = mat_ - mat_gt

    l2 = wp.ddot(mat_diff, mat_diff)
    # l2 = wp.sqrt(
    #     mat_diff[0, 0] ** 2.0
    #     + mat_diff[0, 1] ** 2.0
    #     + mat_diff[0, 2] ** 2.0
    #     + mat_diff[1, 0] ** 2.0
    #     + mat_diff[1, 1] ** 2.0
    #     + mat_diff[1, 2] ** 2.0
    #     + mat_diff[2, 0] ** 2.0
    #     + mat_diff[2, 1] ** 2.0
    #     + mat_diff[2, 2] ** 2.0
    # )

    wp.atomic_add(loss, 0, l2)


@wp.kernel
def compute_Closs_with_grad(
    mpm_state: MPMStateStruct,
    gt_mat: wp.array(dtype=wp.mat33),
    grad: wp.array(dtype=wp.mat33),
    dt: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    mat_ = mpm_state.particle_C[tid]
    mat_gt = gt_mat[tid]

    mat_gt = mat_gt - grad[tid] * dt
    # l1_diff = wp.abs(pos - pos_gt)

    mat_diff = mat_ - mat_gt
    l2 = wp.ddot(mat_diff, mat_diff)

    wp.atomic_add(loss, 0, l2)
