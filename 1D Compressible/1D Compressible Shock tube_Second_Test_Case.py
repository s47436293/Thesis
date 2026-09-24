# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 06:21:16 2026

@author: James Krek
"""


import numpy as np
import matplotlib.pyplot as plt

#Physical Constants
gamma_g  = 1.4 #Heat-Capcity ratio
rg_phys = 287.0 #Specific as constant for air [J/Kg/K]
mu_phys = 1.0e-5 #Dynamic Viscosity [Kg/m/s]
Pr       = 0.71 #Prandtl numbe
 
T_r     = 2400.0 #Reference Temperature for nondimensionilastion of dt [k]
T_inf   = 300.0 #Free-strgeam temperature [k]
p_inf = 101325.0 #Free-stream pressure [Pa]
rho_inf = p_inf / (rg_phys * T_inf)    # Free-stream density [kg/m3]
 
#Domain Discretisation
L_phys = 1.0 #Channel Domain Length [m]
NX = 401 #Nodes
dx_phys = L_phys / (NX-1) #Delta X in Physical Units [m]
dt_phys = dx_phys / np.sqrt(3.0 * rg_phys * T_r) #Delta T in Physical Units [s]
Cu      = dx_phys / dt_phys # Velocity unit [m/s]
 
NSTEPS = 400 #Time Steps
teval     = 50 # Print interval
PLOT_XLIM = (0.0, 1.0) #
 
 
#Lattice Constants (Dimensionless)
cs2 = 1.0 / 3.0 # D1Q3 sound speed square
rg  = cs2 # Lattice gas constant  (= cs2)
cv  = rg / (gamma_g - 1.0) # Specific heat, const. volume      (4.2.38)
cp  = gamma_g * rg / (gamma_g - 1.0)# Specific heat, const. presure (4.2.38)
dx  = 1.0
dt  = 1.0
 
theta_inf = T_inf / T_r #Dimensionless Temperature
mu_latt   = mu_phys * dt_phys / (rho_inf * dx_phys**2) #Dimensionless Viscosity
 
#D1Q3 Lattice
NPOP = 3
cx   = np.array([0, 1, -1], dtype=int)
w    = np.array([2.0/3.0, 1.0/6.0, 1.0/6.0]) #Weight values
Hermite = np.array([np.ones(3), cx, cx**2 - cs2]) # Hermite Polynomials H_{0,0} = Hermite[0,0], H_{0,x} = Hermite[1,0] etc
proj = w * Hermite[2]/ (2.0 * cs2**2)# w_i H_xx /(2 cs^4) = (-1, 1/2, 1/2)
 
 
#Numerical Parameters
D_phys = 1
kappa = (D_phys+2)/D_phys
sigma = 0.05
eps   = 1.0e-14 # guard for the limiter ratio
 
 
#Initial Conditions for Sod Shock tube
 
# Right state is the free stream (rho_inf, p_inf, T_inf).
# Left state: 8x density, 10x pressure  ->  T_L = (10/8) T_inf = 375 K
rho_L, rho_R     = 8.0, 1.0 #Density Left and Right
T_L_phys         = (10.0/8.0) * T_inf       # 375 K
theta_L, theta_R = T_L_phys / T_r, theta_inf #Temperature Left and Right
x_latt = np.arange(NX, dtype=float)
x_norm = x_latt / (NX - 1)
 
delta = 2.0 # tanh smoothing over ~2 cells
ramp  = 0.5 * (1.0 - np.tanh((x_latt - 0.5*(NX-1)) / delta))
 
rho   = rho_R   + (rho_L   - rho_R)   * ramp
theta = theta_R + (theta_L - theta_R) * ramp
u     = np.zeros(NX)
 
#STABILITY CHECK  (Eqs. 4.3.38 - 4.3.39)
Ma_s  = 1.6556                     # from the exact solution for p_L/p_R = 10
c_max = np.sqrt(gamma_g * cs2 * theta_L)
CFL   = (Ma_s + 1.0) * c_max
theta_max_allowed = 3.0 / (gamma_g * (Ma_s + 1.0)**2)
print(f"  Ma_shock ~ {Ma_s:.3f}  |  c* = {c_max:.4f}  |  CFL ~ {CFL:.4f}")
print(f"  theta_L   = {theta_L:.4f}  <  theta_max = {theta_max_allowed:.4f}   "
      f"-> {'OK' if CFL < 1.0 else 'UNSTABLE'}")
print("=" * 62 + "\n")
 
 
def calc_density(fi):
    """
    Parameters
    ----------
    fi : numpy array Shape [N,3]
        Particle distribution function.
 
    Returns
    -------
    rho : numpy array shape [N]
        Density of particles as each node.
 
    """
    rho = np.sum(fi, axis=-1)
    return rho
 
 
def calc_velocity(fi, rho):
    """
    Parameters
    ----------
    fi : numpy array shape [N,3]
        Particle distribution function..
    rho : numpy array shape [N]
        Density of particles as each node.
 
    Returns
    -------
    u : numpy array shape [N]
        Velocity of Particles at each node.
    """
    u = np.sum(fi * cx[None,:], axis=-1) / rho   # FIX 1: "axies" -> "axis"
    return u
 
 
def calc_feq(rho, u , theta):
    """Equilibrium, Eq. (4.2.7) truncated at N = 2. Exact for the first three moments."""
    feq = np.empty((NX,NPOP))
    feq[:,0] = w[0] * rho * (1 - u**2 * 0.5 / cs2 - 0.5 * (theta-1))
    feq[:,1] = w[1] * rho * (1 + (u/cs2) + (u**2/ (2*cs2**2 ) - (u**2/ (2 * cs2)) + (theta-1) * (1-cs2)/(2*cs2)) )
    feq[:,2] = w[2] * rho * (1 - (u/cs2) + (u**2/ (2*cs2**2 ) - (u**2/ (2 * cs2)) + (theta-1) * (1-cs2)/(2*cs2)) )
    return feq
 
 
def ddx(f):
    """Second-order centred difference, Eq. (4.3.28), dx = 1."""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2.0*dx)
 
 
def calc_psi(rho, u, theta):
    """Correction term, Eqs. (4.3.10) - (4.3.13). Returns shape [NX, 3]."""
    p_loc = rho * cs2 * theta #Local Pressure
    Gamma = rho * u * (1- theta - u**2)
    sgn = np.sign(u) #Sign of the Velocity
    dGb = (Gamma - np.roll(Gamma, 1)) / dx              # backward
    dGf = (np.roll(Gamma, -1) - Gamma) / dx             # FIX 3: was Gamma - roll(-1)
    E1 = 0.5 * (1+ sgn) * dGb + 0.5 * (1-sgn) * dGf     # FIX 2: was "*" between branches
    E2 = p_loc * (kappa - gamma_g) * ddx(u)
    return (E1 + E2)[:, None] * proj[None, :]           # FIX 4: outer product
 
 
def a1_fd_calc(rho, theta, u, tau_chk):
    """Eq. (4.3.27). -(tau~/tau) mu (2 - 2/D) du/dx, with (tau~/tau) mu = tau~ p."""
    p_loc = rho * cs2 * theta
    return -tau_chk * p_loc * (2 - 2/D_phys) * ddx(u)   # FIX 5: leading minus
 
 
def a1_pr_calc(g, feq, psi):
    """Eq. (4.3.26). g MUST be the streamed populations at level n+1."""
    placeholder = Hermite[2] * (g - feq + 0.5*dt * psi)
    return np.sum(placeholder, axis = -1)
 
 
def f_stream(f):
    f_out = np.empty_like(f)
    for k in range(NPOP):
        f_out[:, k] = np.roll(f[:, k], cx[k])
    return f_out
 
 
def advect_entropy(s, u):
    """MUSCL convection, Eqs. (4.3.33) - (4.3.36)."""
    chi = 1/3
    deltas_ph  = np.roll(s, -1) - s        # delta s_{i+1/2}
    deltas_mh  = np.roll(deltas_ph,  1)    # delta s_{i-1/2}
    deltas_3ph = np.roll(deltas_ph, -1)    # delta s_{i+3/2}
    deltas_3mh = np.roll(deltas_ph,  2)    # delta s_{i-3/2}
 
    den = np.where(np.abs(deltas_ph) < eps, eps, deltas_ph)   # guard the magnitude
    r   = deltas_mh / den
    phi = np.maximum(0.0, 2*r / (1 + r**2 + eps))   # van Albada (renamed from psi)
 
    # S_{i + 1/2} equations Left and Right
    SL_ph = s + phi/4 * ((1-chi) * deltas_mh + (1+chi) * deltas_ph)
    SR_ph = np.roll(s,-1) - np.roll(phi,-1)/4 * ((1-chi)* deltas_3ph + (1+chi) * deltas_ph)
 
    # S_{i - 1/2} equations Left and Right
    SL_mh = np.roll(s, 1) + np.roll(phi,1)/ 4 * ((1- chi) * deltas_3mh + (1+chi) * deltas_mh )
    SR_mh = s  - phi/ 4 * ((1- chi) * deltas_ph + (1+chi) * deltas_mh)
 
    S_ph = np.where(u >= 0.0, SL_ph, SR_ph)
    S_mh = np.where(np.roll(u,1) >= 0.0, SL_mh, SR_mh)   # FIX 6: face i-1/2 uses u_{i-1}
 
    return u * (S_ph - S_mh)/ dx
 
 
def entropy_RHS(rho, u, theta, s, a1, tau, tau_chk):
    """RHS of Eq. (4.3.32)."""
    conv = advect_entropy(s, u)
    lam  = mu_latt * cp / Pr
    diff_fourier = lam * (np.roll(theta, -1) - 2*theta + np.roll(theta, 1))/dx**2
    Phi = -(tau / tau_chk) * a1 * ddx(u)                 # Eq. (4.3.37)
    return -conv + (diff_fourier + Phi) / (rho * theta)  # FIX 7: both source terms +ve
 
 
def s_from_state(rho, theta):
    """Gibbs relation, Eq. (4.2.6)."""
    return cv * np.log(theta / rho**(gamma_g - 1.0))
 
 
def theta_from_s(rho, s):
    """Inverse Gibbs, Eq. (D17)."""
    return rho**(gamma_g - 1.0) * np.exp(s / cv)
 
 
def update(rho, u, theta, s, a1, feq, psi):
    """One time step - Algorithm 1 p. 131. FIX 8: single definition, with streaming."""
    #Constants for equillibirum update
    p_loc   = rho * cs2 * theta
    tau     = mu_latt / p_loc
    tau_chk = tau + 0.5*dt # tau tilde, change of variable
    omega   = dt / tau_chk
 
    #Compute gi+ Collision step (line 5)
    g1  = a1[:, None] * proj[None, :]
    gi_plus = feq + (1- omega)[:, None]*g1 + 0.5 * dt * psi
 
    #STREAM, then density and velocity for n+1 (line 6)
    gi_stream = f_stream(gi_plus)
    rho_nplus = calc_density(gi_stream)
    u_nplus   = calc_velocity(gi_stream, rho_nplus)
 
    #Entropy for state n+1 (line 7). Every input is level n, so this is
    #independent of line 6 - but the listing order is kept for traceability.
    s_nplus = s + dt * entropy_RHS(rho, u, theta, s, a1, tau, tau_chk)
 
    #Compute theta for state n+1 (line 8)
    theta_nplus = theta_from_s(rho_nplus, s_nplus)
 
    #Boundaries MUST be applied here, before feq and psi are built from these
    #fields - otherwise the corrected edge values never reach the next collision.
    rho_nplus, u_nplus, theta_nplus, s_nplus = apply_bc(
        rho_nplus, u_nplus, theta_nplus, s_nplus)
 
    #Compute psi and feq for state n+1 (lines 9, 10)
    psi_nplus = calc_psi(rho_nplus, u_nplus, theta_nplus)
    feq_nplus = calc_feq(rho_nplus, u_nplus, theta_nplus)
 
    #Compute a1xx term (line 11)
    tau_nplus     = mu_latt/ (rho_nplus * cs2 * theta_nplus)
    tau_chk_nplus = tau_nplus + 0.5*dt
    a1_pr_nplus   = a1_pr_calc(gi_stream, feq_nplus, psi_nplus)     # streamed array
    a1_fd_nplus   = a1_fd_calc(rho_nplus, theta_nplus, u_nplus, tau_chk_nplus)
    a1_nplus      = sigma*a1_pr_nplus + (1.0-sigma)*a1_fd_nplus     # Eq. (4.3.25)
    apply_bc(a1_nplus)
 
    return rho_nplus, u_nplus, theta_nplus, s_nplus, a1_nplus, feq_nplus, psi_nplus
 
 
def apply_bc(*arrays):
    """Zero-gradient outflow, applied to every array passed in."""
    for arr in arrays:
        arr[0]  = arr[1]
        arr[-1] = arr[-2]
    return arrays
 
 
#Initialise (Algorithm lines 1 - 4) and run
s   = s_from_state(rho, theta)
tau_chk0 = mu_latt/(rho*cs2*theta) + 0.5*dt
a1  = a1_fd_calc(rho, theta, u, tau_chk0)
feq = calc_feq(rho, u, theta)
psi = calc_psi(rho, u, theta)
 
for step in range(NSTEPS):
    rho, u, theta, s, a1, feq, psi = update(rho, u, theta, s, a1, feq, psi)
    if not (np.all(np.isfinite(rho)) and np.all(rho > 0) and np.all(theta > 0)):
        raise FloatingPointError(f"diverged at step {step}")
    if step % teval == 0:
        c_loc = np.sqrt(gamma_g * cs2 * theta)
        print(f"  Step {step:4d}  |  rho [{rho.min():.4f}, {rho.max():.4f}]  "
              f"theta [{theta.min():.5f}, {theta.max():.5f}]  "
              f"Ma_max {np.max(np.abs(u)/c_loc):.3f}")
 
print("\nSimulation complete - building plot.")
 
 
#Exact Riemann solution (Toro) for the analytic overlay
def sod_analytic(rho_L, p_L, u_L, rho_R, p_R, u_R, gamma, x_grid, x0, t):
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    G   = (gamma - 1.0) / (gamma + 1.0)
 
    def fw(p, pK, rK, cK):
        if p > pK:
            A = 2.0 / ((gamma + 1.0)*rK)
            return (p - pK) * np.sqrt(A / (p + G*pK))
        return 2.0*cK/(gamma - 1.0) * ((p/pK)**((gamma - 1.0)/(2.0*gamma)) - 1.0)
 
    def F(p):
        return fw(p, p_L, rho_L, c_L) + fw(p, p_R, rho_R, c_R) + u_R - u_L
 
    lo, hi = 1.0e-12, 10.0*max(p_L, p_R)
    for _ in range(300):
        mid = 0.5*(lo + hi)
        if F(lo)*F(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0e-14:
            break
    ps = 0.5*(lo + hi)
    us = 0.5*(u_L + u_R) + 0.5*(fw(ps, p_R, rho_R, c_R) - fw(ps, p_L, rho_L, c_L))
    rsL = rho_L*(ps/p_L + G)/(G*ps/p_L + 1) if ps > p_L else rho_L*(ps/p_L)**(1/gamma)
    rsR = rho_R*(ps/p_R + G)/(G*ps/p_R + 1) if ps > p_R else rho_R*(ps/p_R)**(1/gamma)
 
    xi = (x_grid - x0) / t
    rho_o = np.empty_like(x_grid); u_o = np.empty_like(x_grid); p_o = np.empty_like(x_grid)
    for i, sx in enumerate(xi):
        if sx < us:
            if ps <= p_L:
                csL = c_L*(ps/p_L)**((gamma - 1)/(2*gamma))
                if sx < u_L - c_L:
                    rho_o[i], u_o[i], p_o[i] = rho_L, u_L, p_L
                elif sx < us - csL:
                    ul = 2/(gamma + 1)*(c_L + (gamma - 1)/2*u_L + sx)
                    cl = c_L - (gamma - 1)/2*(ul - u_L)
                    rho_o[i] = rho_L*(cl/c_L)**(2/(gamma - 1)); u_o[i] = ul
                    p_o[i]   = p_L*(cl/c_L)**(2*gamma/(gamma - 1))
                else:
                    rho_o[i], u_o[i], p_o[i] = rsL, us, ps
            else:
                SL = u_L - c_L*np.sqrt((gamma+1)/(2*gamma)*ps/p_L + (gamma-1)/(2*gamma))
                rho_o[i], u_o[i], p_o[i] = (rho_L, u_L, p_L) if sx < SL else (rsL, us, ps)
        else:
            if ps <= p_R:
                csR = c_R*(ps/p_R)**((gamma - 1)/(2*gamma))
                if sx > u_R + c_R:
                    rho_o[i], u_o[i], p_o[i] = rho_R, u_R, p_R
                elif sx > us + csR:
                    ur = 2/(gamma + 1)*(-c_R + (gamma - 1)/2*u_R + sx)
                    cr = c_R + (gamma - 1)/2*(u_R - ur)
                    rho_o[i] = rho_R*(cr/c_R)**(2/(gamma - 1)); u_o[i] = ur
                    p_o[i]   = p_R*(cr/c_R)**(2*gamma/(gamma - 1))
                else:
                    rho_o[i], u_o[i], p_o[i] = rsR, us, ps
            else:
                SR = u_R + c_R*np.sqrt((gamma+1)/(2*gamma)*ps/p_R + (gamma-1)/(2*gamma))
                rho_o[i], u_o[i], p_o[i] = (rsR, us, ps) if sx < SR else (rho_R, u_R, p_R)
    return rho_o, u_o, p_o
 
 
#Analytic reference and normalisation
c_R_phys = np.sqrt(gamma_g * rg_phys * T_inf)
rho_a, u_a, p_a = sod_analytic(8.0*rho_inf, 10.0*p_inf, 0.0, rho_inf, p_inf, 0.0,
                               gamma_g, x_norm*L_phys, 0.5*L_phys, NSTEPS*dt_phys)
T_a = p_a / (rho_a * rg_phys)
 
rho_norm_sim = rho / rho_L;             rho_norm_an = rho_a / (8.0*rho_inf)
T_norm_sim   = theta * T_r / T_L_phys;  T_norm_an   = T_a / T_L_phys
u_norm_sim   = u * Cu / c_R_phys;       u_norm_an   = u_a / c_R_phys
 
for name, a, b in (("rho", rho_norm_sim, rho_norm_an),
                   ("T",   T_norm_sim,   T_norm_an),
                   ("u",   u_norm_sim,   u_norm_an)):
    print(f"  normalised L1 error in {name:3s}: "
          f"{np.linalg.norm(a-b,1)/NX/max(np.ptp(b),1e-12):.4f}")
 
 
#Plot
fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.5), sharex=True)
stride = 8
for ax, an, sim, ylabel in zip(axes,
        [rho_norm_an, T_norm_an, u_norm_an],
        [rho_norm_sim, T_norm_sim, u_norm_sim],
        [r"$\rho\,/\,\rho_L$", r"$T\,/\,T_L$", r"$u\,/\,c_R$"]):
    ax.plot(x_norm, an, "k-", lw=1.4, label="Analytic")
    ax.plot(x_norm[::stride], sim[::stride], "o", color="tab:red",
            mfc="none", ms=5, lw=0, label="Present (HLBM)")
    ax.plot(x_norm, sim, ":", color="tab:red", lw=0.9)
    ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
axes[0].legend(loc="lower left", fontsize=9)
axes[2].set_xlabel(r"$x\;/\;L$"); axes[2].set_xlim(PLOT_XLIM)
fig.suptitle(rf"D1Q3 HLBM - $\kappa={kappa:.3f}$, $\sigma={sigma}$, "
             rf"$N={NX}$, step {NSTEPS}", fontsize=9)
fig.tight_layout()
plt.show()
    