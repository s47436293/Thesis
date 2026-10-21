# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:08:34 2026

@author: krekj
"""


import os
import sys

# Finds the 'engg4600' parent directory dynamically relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../compressible_2d
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) # .../engg4600

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Global imports now work seamlessly anywhere
from ThesisCode.Compressible_1D.Analytical_Sod_Shock_Solver import sod_analytical_solver

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pyvista as pv

#Physical Constants
gamma_g  = 1.4 #Heat-Capacity ratio
rg_phys = 287.0 #Specific gas constant for air [J/Kg/K]
mu_phys = 1.0e-5 #Dynamic Viscosity [Kg/m/s]
Pr       = 0.71 #Prandtl number
T_r     = 1460.0 #Reference Temperature for nondimensionalisation of dt [K]
T_inf   = 300.0 #Free-stream temperature [K]
p_inf = 101325.0 #Free-stream pressure [Pa]
rho_inf = p_inf / (rg_phys * T_inf)    # Free-stream density [kg/m3]


#Domain Discretisation (base grid, 401 nodes)
L_phys = 1.0 #Channel Domain Length [m]
NX = 401 #Nodes
dx_phys = L_phys / (NX-1) #Delta X in Physical Units [m]
dt_phys = dx_phys / np.sqrt(3.0 * rg_phys * T_r) #Delta T in Physical Units [s]
Cu      = dx_phys / dt_phys # Velocity unit [m/s] (same for every grid: sqrt(3 R T_r))

NSTEPS = 300 #Time Steps on the 401 grid
t_out  = NSTEPS * dt_phys #Physical output time [s], shared by all grids
PLOT_XLIM = (0.0, 1.0)


#Lattice Constants (Dimensionless)
theta_inf = T_inf / T_r #Dimensionless Temperature
c_R_phys = np.sqrt(gamma_g * rg_phys * T_inf)


#Initial Conditions for Strong Shock tube
rho_L, rho_R     = 10.0, 1.0 #Density Left and Right (lattice units)
theta_L, theta_R = theta_inf, theta_inf #Temperature Left and Right


plt.style.use('science')

#Analytical Solution (fine grid so the jumps are drawn sharply)
x_norm_an = np.linspace(0.0, 1.0, 2001)
rho_a, u_a, p_a = sod_analytical_solver(rho_L*rho_inf, rho_L*p_inf, 0.0, rho_inf, p_inf, 0.0, gamma_g, x_norm_an*L_phys, 0.5*L_phys, t_out)

#Normalise Analytical For plot
T_a = p_a / (rho_a * rg_phys)
rho_norm_an = rho_a / rho_inf
T_norm_an   = T_a / T_inf
u_norm_an   = u_a / c_R_phys


#Import 1D TCLB CURVES

def x_nodes(nx):
    #Node positions x_i = i*dx with dx = L/(nx-1), normalised by L
    return np.arange(nx) / (nx - 1)


def steps_for(nx):
    #Iterations needed to reach t_out: dt scales with dx, so steps scale with (nx-1)
    return NSTEPS * (nx - 1) // (NX - 1)




#Base Folder Where 1D Vtis exist
vti_Folder_1D = os.path.join(current_dir, 'TCLB_1D_Output_Vtis')
SOD_401 = os.path.join(vti_Folder_1D, 'StrongShockd1q3_sod_Rho,U,Theta,Entropy_P00_00000300.pvti')
SOD_801 = os.path.join(vti_Folder_1D, 'StrongShock801d1q3_sod_Rho,U,Theta,Entropy_P00_00000600.pvti')
SOD_1601 = os.path.join(vti_Folder_1D, 'StrongShock1601d1q3_sod_Rho,U,Theta,Entropy_P00_00001200.pvti')



def extract_data(file, nx):
    data = pv.read(file)
    rho   = np.asarray(data["Rho"]) / rho_R
    theta = np.asarray(data["Theta"]) / theta_inf
    u     = np.asarray(data["U"])[:, 0] * Cu / c_R_phys

    #Warn instead of silently plotting nothing
    bad = ~np.isfinite(rho) | (rho <= 0)
    if bad.any():
        print(f"WARNING {nx} nodes: {bad.sum()} NaN/non-positive nodes, "
              f"first at x/L = {np.argmax(bad)/(nx-1):.3f} -> run diverged")
    return rho, theta, u

rho401,  theta401,  u401  = extract_data(SOD_401, 401)
rho801,  theta801,  u801  = extract_data(SOD_801, 801)
rho1601, theta1601, u1601 = extract_data(SOD_1601, 1601)

x_norm_401  = x_nodes(401)
x_norm_801  = x_nodes(801)
x_norm_1601 = x_nodes(1601)

with plt.style.context(["science","no-latex" ]):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.5, 8.5),sharex = True )

    #Density
    ax1.plot(x_norm_an, rho_norm_an, "k-", lw= 1.4, label = "Analytic")
    ax1.plot(x_norm_401, rho401, ls=":", color="tab:red", marker="^", markevery=10, mfc="none", lw=0.9, label="401")
    ax1.plot(x_norm_801, rho801, ls=":", color="tab:blue", marker="o", markevery=20, mfc="none", lw=0.9, label="801")
    ax1.plot(x_norm_1601, rho1601, ls=":", color="tab:green", marker="s", markevery=40, mfc="none", lw=0.9, label="1601")
    ax1.set_ylabel(r"$\rho\,/\,\rho_R$")
    ax1.grid(alpha=0.25)

    #Temperature
    ax2.plot(x_norm_an, T_norm_an, "k-", lw= 1.4)
    ax2.plot(x_norm_401, theta401, ls=":", color="tab:red", marker="^", markevery=10, mfc="none", lw=0.9)
    ax2.plot(x_norm_801, theta801, ls=":", color="tab:blue", marker="o", markevery=20, mfc="none", lw=0.9)
    ax2.plot(x_norm_1601, theta1601, ls=":", color="tab:green", marker="s", markevery=40, mfc="none", lw=0.9)
    ax2.set_ylabel(r"$T\,/\,T_L$")
    ax2.grid(alpha=0.25)

    #Velocity
    ax3.plot(x_norm_an, u_norm_an, "k-", lw= 1.4)
    ax3.plot(x_norm_401, u401, ls=":", color="tab:red", marker="^", markevery=10, mfc="none", lw=0.9)
    ax3.plot(x_norm_801, u801, ls=":", color="tab:blue", marker="o", markevery=20, mfc="none", lw=0.9)
    ax3.plot(x_norm_1601, u1601, ls=":", color="tab:green", marker="s", markevery=40, mfc="none", lw=0.9)
    ax3.set_ylabel(r"$u\,/\,c_R$")
    ax3.set_xlabel(r"$x\;/\;L$")
    ax3.grid(alpha=0.25)
    ax3.set_xlim(*PLOT_XLIM)

    ax1.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    plt.savefig("StrongShock_TCLB_grid_convergence.png", dpi=300)
    plt.show()

