# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 16:07:20 2026

@author: James Krek
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
import pyvista as pv
from Analytical_Sod_Shock_Solver import sod_analytical_solver
plt.style.use('science')

#Physical Constants 
gamma_g  = 1.4 #Heat-Capcity ratio
rg_phys = 287.0 #Specific as constant for air [J/Kg/K]
mu_phys = 1.0e-5 #Dynamic Viscosity [Kg/m/s]
Pr       = 0.71 #Prandtl numbe

T_r     = 1460.0 #Reference Temperature for nondimensionilastion of dt [k]
T_inf   = 300.0 #Free-strgeam temperature [k]               
p_inf = 101325.0 #Free-stream pressure [Pa]
rho_inf = p_inf / (rg_phys * T_inf)    # Free-stream density [kg/m3]

#Domain Discretisation
L_phys = 1.0 #Channel Domain Length [m]
NX = 401 #Nodes
dx_phys = L_phys / (NX-1) #Delta X in Physical Units [m]
dt_phys = dx_phys / np.sqrt(3.0 * rg_phys * T_r) #Delta T in Physical Units [s]
Cu      = dx_phys / dt_phys # Velocity unit [m/s]

NSTEPS = 358 #Time Stepsth
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
sigma = 0.4


#Initial Conditions for Sod Shock tube

rho_L, rho_R     = 3.0, 1.0 #Density Left and Right
theta_L, theta_R = theta_inf, theta_inf #Temperature Left and Right
x_latt = np.arange(NX, dtype=float)
x_norm = x_latt / (NX - 1)

delta = 2.0 # tanh smoothing over ~2 cells
ramp  = 0.5 * (1.0 - np.tanh((x_latt - 0.5*(NX-1)) / delta))

rho   = rho_R   + (rho_L   - rho_R)   * ramp
theta = theta_R + (theta_L - theta_R) * ramp
u     = np.zeros(NX)

#STABILITY CHECK  (Eqs. 4.3.38 - 4.3.39)
Ma_s  = np.sqrt((3.0*(gamma_g + 1.0) + (gamma_g - 1.0)) / (2.0*gamma_g))
c_max = np.sqrt(gamma_g * cs2 * theta_inf)
CFL   = (Ma_s + 1.0) * c_max
theta_max_allowed = 3.0 / (gamma_g * (Ma_s + 1.0)**2)
print(f"  Ma_shock ~ {Ma_s:.3f}  |  c* = {c_max:.4f}  |  CFL ~ {CFL:.4f}")
print(f"  theta_inf = {theta_inf:.4f}  <  theta_max = {theta_max_allowed:.4f}   "
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
    u = np.sum(fi * cx[None,:], axis = -1)/ rho
    return u

def calc_feq(rho, u , theta):
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
    dGb = (Gamma - np.roll(Gamma, 1)) / dx              
    dGf = (np.roll(Gamma, -1) - Gamma) / dx             
    E1 = 0.5 * (1+ sgn) * dGb + 0.5 * (1-sgn) * dGf     
    E2 = p_loc * (kappa - gamma_g) * ddx(u)
    return (E1 + E2)[:, None] * proj[None, :]           
"""  
def calc_psi(rho, u, theta):
    p_loc = rho * cs2 * theta #Local Pressure 
    Gamma = rho * u * (1 - theta - u**2)
    sgn   = np.sign(u)
    dGb   = (Gamma - np.roll(Gamma, 1)) / dx # backward
    dGf   = (np.roll(Gamma, -1) - Gamma) / dx # forward
    E1    = 0.5*(1 + sgn)*dGb + 0.5*(1 - sgn)*dGf      
    E2    = p_loc * (kappa - gamma_g) * ddx(u)        
    proj  = w * Hermite[2] / (2*cs2**2)                
    return (E1 + E2)[:, None] * proj[None, :]
"""

def a1_fd_calc(rho, theta, u, tau_chk):
    p_loc = rho * cs2 * theta
    return -tau_chk * p_loc * (2- 2/D_phys) * ddx(u)

       
def a1_pr_calc(g, feq, psi):
    
    placeholder = Hermite[2] * (g-feq + 0.5*dt * psi)
    return np.sum(placeholder, axis = -1)


def f_stream(f):
    f_out = np.empty_like(f)
    for k in range(NPOP):
        f_out[:, k] = np.roll(f[:, k], cx[k])
    return f_out


def advect_entropy(s, u):
    chi = 1/3
    deltas_ph  = np.roll(s, -1) - s        # delta s_{i+1/2}
    deltas_mh  = np.roll(deltas_ph,  1)    # delta s_{i-1/2}
    deltas_3ph = np.roll(deltas_ph, -1)    # delta s_{i+3/2}
    deltas_3mh = np.roll(deltas_ph,  2)    # delta s_{i-3/2}
    eps = 1.0e-14
    den = np.where(np.abs(deltas_ph) < eps, eps, deltas_ph)
    r   = deltas_mh / den
    phi = np.maximum(0.0, 2*r / (1 + r**2 + eps))
    
    # S_{i + 1/2} equations Left and Right 
    SL_ph = s + phi/4 * ((1-chi) * deltas_mh + (1+chi) * deltas_ph)
    SR_ph = np.roll(s,-1) - np.roll(phi,-1)/4 * ((1-chi)* deltas_3ph + (1+chi) * deltas_ph)
    
    # S_{i - 1/2} equations Left and Right
    SL_mh = np.roll(s, 1) + np.roll(phi,1)/ 4 * ((1- chi) * deltas_3mh + (1+chi) * deltas_mh )
    SR_mh = s  - phi/ 4 * ((1- chi) * deltas_ph + (1+chi) * deltas_mh)
    
    S_ph = np.where(u>= 0.0, SL_ph, SR_ph)
    S_mh = np.where(np.roll(u,1) >= 0.0, SL_mh, SR_mh)
    return u * (S_ph - S_mh)/ dx


def entropy_RHS(rho, u, theta, s, a1, tau, tau_chk):
    conv = advect_entropy(s, u)
    lam  = mu_latt * cp / Pr
    diff_fourier = (-lam) * (np.roll(theta, -1) - 2* theta + np.roll(theta, 1))/dx**2
    phi = -(tau / tau_chk) * a1 * ddx(u)
    return -conv + (diff_fourier + phi) / (rho * theta)

def s_from_state(rho, theta):
    """Gibbs relation, Eq. (4.2.6)."""
    return cv * np.log(theta / rho**(gamma_g - 1.0))

def theta_from_s(rho, s):
    """Inverse Gibbs, Eq. (D17)."""
    return rho**(gamma_g - 1.0) * np.exp(s / cv)


    
def apply_bc(*arrays):
    """Zero-gradient outflow, applied to every array passed in."""
    for arr in arrays:
        arr[0]  = arr[1]
        arr[-1] = arr[-2]
    return arrays   
    
def update(rho, u, theta, s, a1, feq, psi):
    #Constants for equillibirum update n = 1
    p_loc   = rho * cs2 * theta
    tau     = mu_latt / p_loc              
    tau_chk = tau + 0.5*dt # tau tilde, change of variable
    omega   = dt / tau_chk
    
    #Compute gi+ Collision step and then stream
    g1  = a1[:, None] * proj[None, :]  
    gi_plus = feq + (1- omega)[:, None]*g1 + 0.5 * dt * psi
    gi_stream = f_stream(gi_plus)
    
    #Compute density and velocity for n+1 state
    rho_nplus = calc_density(gi_stream)
    u_nplus = calc_velocity(gi_stream, rho_nplus)
    
    #Compute entropy for state n+1
    s_nplus = s + dt * entropy_RHS(rho, u, theta, s, a1, tau, tau_chk)
    
    #Compute theta for state n+1
    theta_nplus = theta_from_s(rho_nplus, s_nplus)
    
    
    #Boundary Conditions
    rho_nplus, u_nplus, theta_nplus, s_nplus = apply_bc(
        rho_nplus, u_nplus, theta_nplus, s_nplus)
    
    #Compute psi for state n+1
    psi_nplus = calc_psi(rho_nplus, u_nplus, theta_nplus)
    
    #Compute feq for state n+1 
    feq_nplus = calc_feq(rho_nplus, u_nplus, theta_nplus)
    
    
    #Compute a1xx term
    tau_nplus  = mu_latt/ (rho_nplus * cs2 * theta_nplus)
    tau_chk_nplus = tau_nplus + 0.5*dt
    a1_pr_nplus   = a1_pr_calc(gi_stream, feq_nplus, psi_nplus)
    a1_fd_nplus   = a1_fd_calc(rho_nplus, theta_nplus, u_nplus, tau_chk_nplus)
    a1_nplus      = sigma*a1_pr_nplus + (1.0-sigma)*a1_fd_nplus
    apply_bc(a1_nplus)
    return rho_nplus, u_nplus, theta_nplus, s_nplus, a1_nplus, feq_nplus, psi_nplus



 
#Initialise Algorithm
s = s_from_state(rho, theta)
tau_chk = mu_latt/(rho*cs2*theta) + 0.5*dt
a1  = a1_fd_calc(rho, theta, u, tau_chk)
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
 
 


 
#Analytic reference and normalisation
c_R_phys = np.sqrt(gamma_g * rg_phys * T_inf)

rho_a, u_a, p_a = sod_analytical_solver(3.0*rho_inf, 3.0*p_inf, 0.0, rho_inf, p_inf, 0.0, gamma_g, x_norm*L_phys, 0.5*L_phys, NSTEPS*dt_phys)   
    
T_a = p_a / (rho_a * rg_phys)
rho_norm_an = rho_a / rho_inf
T_norm_an   = T_a / T_inf
u_norm_an   = u_a / c_R_phys

#Normalise Simulation Values
rho_norm_sim = rho/ rho_R
T_norm_sim = theta/ theta_inf
u_norm_sim   = u * Cu / c_R_phys
 
for name, a, b in (("rho", rho_norm_sim, rho_norm_an),
                   ("T",   T_norm_sim,   T_norm_an),
                   ("u",   u_norm_sim,   u_norm_an)):
    print(f"  normalised L1 error in {name:3s}: "
          f"{np.linalg.norm(a-b,1)/NX/max(np.ptp(b),1e-12):.4f}")
    
    


#Importing Digitised Curves

#Rho import
rho_norm_digi = pd.read_csv('Florian_Digitised_SodShiock_CSVs\Florian_Digitised_rhoRhor.csv', header=None, names=["x", "y"])
rho_norm_digi = rho_norm_digi.sort_values("x")

#T import
T_norm_digi = pd.read_csv('Florian_Digitised_SodShiock_CSVs\Florian_Digitised_TTL.csv', header=None, names=["x", "y"])
T_norm_digi = T_norm_digi.sort_values("x")


#U import 
U_norm_digi = pd.read_csv('Florian_Digitised_SodShiock_CSVs\Florian_Digitised_uCr.csv', header=None, names=["x", "y"])
U_norm_digi = U_norm_digi.sort_values("x")


#Importing TCLB code profiles
#mesh = pv.read("Florian1d1q3_sod_Rho,U,Theta,Entropy_P00_00000358.vti")


mesh = pv.read("5d2q9sodd2q9_sod_VTK_P00_00000358.vti")
x_norm_TCLB = (np.arange(NX) + 0.5) * dx_phys / L_phys      # cell centres
nx, ny, _ = np.array(mesh.dimensions) - 1     # points -> cells: 401, 4, 1
j  = ny // 2                                  # any row
sl = slice(j*nx, (j+1)*nx)                    # that row, x fastest

rho_TCLB   = mesh["Rho"][sl]
u_TCLB     = mesh["U"][sl]                    # still (nx, 3)
theta_TCLB = mesh["Theta"][sl]
"""
rho_TCLB = mesh["Rho"]
u_TCLB = mesh["U"]
theta_TCLB = mesh["Theta"]

"""

#Normalise TCLB Curves
rho_norm_TCLB = rho_TCLB/ rho_R
theta_norm_TCLB = theta_TCLB/theta_inf
u_norm_TCLB   = u_TCLB[:, 0] * Cu / c_R_phys



#Plot
with plt.style.context(["science", "no-latex"]):
    stride = 8
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.5, 8.5),sharex = True )
    
    #Analytic
   # ax1.plot(x_norm, rho_norm_an, "k-", lw= 1.4, label = "Analytic")
    #Python Script
    """
    ax1.plot(x_norm[::stride], rho_norm_sim[::stride], "o", color ="tab:red", mfc="none", ms=5, lw=0, label="Python (HLBM)")
    ax1.plot(x_norm, rho_norm_sim, ":", color="tab:red", lw=0.9)
    """
    #Florian Thesis
    ax1.plot(rho_norm_digi["x"][::stride], rho_norm_digi["y"][::stride], "o", color ="tab:blue", mfc="none", ms=5, lw=0, label="Florian")
    ax1.plot(rho_norm_digi["x"], rho_norm_digi["y"], ":", color="tab:blue", lw=0.9)
    #TCLB script
    ax1.plot(x_norm_TCLB[::stride], rho_norm_TCLB[::stride], "o", color="tab:green", mfc="none", ms=5, lw=0, label="TCLB (HLBM)")
    ax1.plot(x_norm_TCLB,           rho_norm_TCLB,           ":", color="tab:green", lw=0.9)
    ax1.set_ylabel(r"$\rho\,/\,\rho_R$")
    ax1.grid(alpha=0.25)
 
    
    #Analytic
    #x2.plot(x_norm, T_norm_an, "k-", lw= 1.4, label = "Analytic")
    #Python Script
    """
    ax2.plot(x_norm[::stride], T_norm_sim[::stride], "o", color ="tab:red", mfc="none", ms=5, lw=0, label="Python (HLBM)")
    ax2.plot(x_norm, T_norm_sim, ":", color="tab:red", lw=0.9)
    """
    #Florian thesis
    ax2.plot(T_norm_digi["x"][::stride], T_norm_digi["y"][::stride], "o", color ="tab:blue", mfc="none", ms=5, lw=0, label="Florian")
    ax2.plot(T_norm_digi["x"], T_norm_digi["y"], ":", color="tab:blue", lw=0.9)
    #TCLB solver
    ax2.plot(x_norm_TCLB[::stride], theta_norm_TCLB[::stride], "o", color ="tab:green", mfc="none", ms=5, lw=0, label="TCLB (HLBM)")
    ax2.plot(x_norm_TCLB, theta_norm_TCLB, ":", color="tab:green", lw=0.9)
    ax2.set_ylabel(r"$T\,/\,T_L$")
    ax2.grid(alpha=0.25)
    
    #Analytic
    #ax3.plot(x_norm, u_norm_an, "k-", lw= 1.4, label = "Analytic")
    #Python Script
    """
    ax3.plot(x_norm[::stride], u_norm_sim[::stride], "o", color ="tab:red", mfc="none", ms=5, lw=0, label="Python (HLBM)")
    ax3.plot(x_norm, u_norm_sim, ":", color="tab:red", lw=0.9)
    """
    #Florian Thesis
    ax3.plot(U_norm_digi["x"][::stride], U_norm_digi["y"][::stride], "o", color ="tab:blue", mfc="none", ms=5, lw=0, label="Florian")
    ax3.plot(U_norm_digi["x"], U_norm_digi["y"], ":", color="tab:blue", lw=0.9)
    #TCLB solver
    ax3.plot(x_norm_TCLB[::stride], u_norm_TCLB[::stride], "o", color ="tab:green", mfc="none", ms=5, lw=0, label="TCLB (HLBM)")
    ax3.plot(x_norm_TCLB, u_norm_TCLB, ":", color="tab:green", lw=0.9)
    ax3.set_ylabel(r"$u\,/\,c_R$")
    ax3.grid(alpha=0.25)
    
#fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.5), sharex=True)

    ax1.legend(loc="lower left", fontsize=9)
    ax3.set_xlabel(r"$x\;/\;L$"); 
    ax3.set_xlim(PLOT_XLIM)
    fig.suptitle(rf"D1Q3 HLBM - $\kappa={kappa:.3f}$, $\sigma={sigma}$, "
                 rf"$N={NX}$, step {NSTEPS}", fontsize=9)
    fig.tight_layout()
    plt.show()
    
    
    
 
    
    