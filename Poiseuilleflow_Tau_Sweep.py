# -*- coding: utf-8 -*-
"""
LBM D2Q9 Poiseuille flow with tau (relaxation time) sweep.
Compares simulated centreline velocity profiles against the analytical
parabolic solution for a range of tau values, and plots L2 error vs tau.

The "magic" tau = sqrt(3/16) + 0.5 gives the exact Poiseuille solution
for D2Q9 BGK with halfway bounce-back; error grows on either side.
"""

import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# Grid and base parameters (fixed across the tau sweep)
# =====================================================================
scale  = 3
NX     = 5 * scale          # streamwise extent
NY     = 5 * scale          # wall-normal extent
NSTEPS = int(1e4 * scale**2)
u_max  = 0.1 / scale        # target centreline velocity (held fixed)

tol   = 1e-12
teval = 100


# =====================================================================
# D2Q9 lattice
# =====================================================================
NPOP = 9
cx = np.array([ 1,  0, -1,  0,  1, -1, -1,  1,  0], dtype=int)
cy = np.array([ 0,  1,  0, -1,  1,  1, -1, -1,  0], dtype=int)
w  = np.array([1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36, 4/9])

U_pos = np.array([0, 4, 7]);  U_neg = np.array([2, 5, 6])
V_pos = np.array([1, 4, 5]);  V_neg = np.array([3, 6, 7])

TOP_IN  = np.array([1, 4, 5]);  TOP_OUT = np.array([3, 6, 7])
BOT_IN  = np.array([3, 6, 7]);  BOT_OUT = np.array([1, 4, 5])


# =====================================================================
# Analytical Poiseuille profile (depends only on u_max and NY)
# =====================================================================
y = np.arange(NY) + 0.5
u_analy = -4 * u_max / NY**2 * (y - 0) * (y - NY)


# =====================================================================
# Core LBM operators
# =====================================================================
def rho_calc(f):
    return np.sum(f, axis=-1)

def vel_calc(f):
    u = np.sum(f[:, :, U_pos], axis=-1) - np.sum(f[:, :, U_neg], axis=-1)
    v = np.sum(f[:, :, V_pos], axis=-1) - np.sum(f[:, :, V_neg], axis=-1)
    return u, v

def feq_calc(rho, u, v):
    return w * (rho[..., np.newaxis]
                + 3 * (u[..., np.newaxis] * cx + v[..., np.newaxis] * cy))

def f_stream(f):
    out = np.empty_like(f)
    for k in range(NPOP):
        out[:, :, k] = np.roll(np.roll(f[:, :, k], cx[k], axis=0), cy[k], axis=1)
    return out


def update(f, omega, rho_inlet, rho_outlet):
    """One LBM step: collide, apply pressure-driven BCs, stream, bounce-back."""
    rho = rho_calc(f)
    u, v = vel_calc(f)
    feq = feq_calc(rho, u, v)

    # Collision
    fstar = (1 - omega) * f + omega * feq

    # Inlet PBBC (ghost node at i = 0)
    u_N = u[NX - 2];  v_N = v[NX - 2]
    rho_in = np.full(NY, rho_inlet)
    feq_in = feq_calc(rho_in, u_N, v_N)
    fstar[0, :, :] = feq_in + (fstar[NX - 2, :, :] - feq[NX - 2, :, :])

    # Outlet PBBC
    u_0 = u[1];  v_0 = v[1]
    rho_out = np.full(NY, rho_outlet)
    feq_out = feq_calc(rho_out, u_0, v_0)
    fstar[-1, :, :] = feq_out + (fstar[1, :, :] - feq[1, :, :])

    # Streaming
    fprop = f_stream(fstar)

    # No-slip walls (halfway bounce-back)
    fprop[:, NY - 1, TOP_OUT] = fstar[:, NY - 1, TOP_IN]
    fprop[:, 0,      BOT_OUT] = fstar[:, 0,      BOT_IN]

    return fprop, u


# =====================================================================
# Single simulation at a given tau
# =====================================================================
def run_sim(tau):
    omega      = 1.0 / tau
    nu         = (2 * tau - 1) / 6
    Re         = NY * u_max / nu
    gradP      = 8 * nu * u_max / NY**2
    rho_outlet = 1.0
    rho_inlet  = 3 * (NX - 1) * gradP + rho_outlet

    f = np.zeros((NX, NY, NPOP))
    for k in range(NPOP):
        f[:, :, k] = w[k]
    fprop = f.copy()
    u_old = np.zeros((NX, NY))

    for step in range(NSTEPS):
        fprop, u = update(fprop, omega, rho_inlet, rho_outlet)
        if (step + 1) % teval == 0:
            conv = abs(u.mean() / (u_old.mean() + 1e-30) - 1)
            if conv < tol:
                break
            u_old = u.copy()

    u_f, v_f = vel_calc(fprop)
    return u_f, v_f, step + 1, Re, nu


# =====================================================================
# Main: sweep tau, plot profiles and error curve
# =====================================================================
def main():
    tau_magic = np.sqrt(3 / 16) + 0.5

    # --- Sweep 1: four selected tau for the profile plot ---
    profile_taus = np.array([0.51, tau_magic, 2.00, 3.00])
    profiles = []

    print("Profile sweep:")
    print(f"{'tau':>8} {'Re':>10} {'steps':>8} {'L2 err':>12}")
    print("-" * 42)
    for tau in profile_taus:
        u_f, _, n_steps, Re, _ = run_sim(tau)
        profile = u_f[NX // 2, :]
        err = np.sqrt(np.mean((profile - u_analy) ** 2)) / u_max
        if not np.isfinite(err):
            err = np.nan
        profiles.append(profile)
        print(f"{tau:8.4f} {Re:10.2f} {n_steps:8d} {err:12.3e}")

    # --- Sweep 2: dense linspace for the error-vs-tau plot ---
    err_taus = np.linspace(0.51, 3.0, 30)
    errors = []

    print("\nError sweep:")
    print(f"{'tau':>8} {'Re':>10} {'steps':>8} {'L2 err':>12}")
    print("-" * 42)
    for tau in err_taus:
        u_f, _, n_steps, Re, _ = run_sim(tau)
        profile = u_f[NX // 2, :]
        err = np.sqrt(np.mean((profile - u_analy) ** 2)) / u_max
        if not np.isfinite(err):
            err = np.nan
        errors.append(err)
        print(f"{tau:8.4f} {Re:10.2f} {n_steps:8d} {err:12.3e}")

    # ---- Plot 1: velocity profiles for the four selected tau ----
    plt.figure(figsize=(7, 5))
    for tau, prof in zip(profile_taus, profiles):
        plt.plot(prof, y, "o-", lw=1, ms=4, label=fr"$\tau={tau:.3f}$")
    plt.plot(u_analy, y, "k-", lw=2, label="Analytical")
    plt.xlabel(r"$u_x$");  plt.ylabel(r"$y$")
    plt.title(r"Centreline velocity profiles for varying $\tau$")
    plt.legend(fontsize=8);  plt.grid(alpha=0.3);  plt.tight_layout()
    plt.show()

    # ---- Plot 2: L2 error vs tau (dense sweep) ----
    plt.figure(figsize=(7, 5))
    plt.semilogy(err_taus, errors, "o-", color="steelblue", ms=4)
    plt.axvline(tau_magic, color="crimson", ls="--", lw=1,
                label=fr"$\tau^* = \sqrt{{3/16}} + 1/2 \approx {tau_magic:.4f}$")
    plt.xlabel(r"$\tau$");  plt.ylabel(r"Normalised $L_2$ error")
    plt.title(r"Error vs relaxation time $\tau$")
    plt.legend();  plt.grid(alpha=0.3, which="both");  plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()