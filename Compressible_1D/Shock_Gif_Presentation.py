# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:44:12 2026

@author: krekj
"""
import os
import sys
import io
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pyvista as pv
from PIL import Image

# Finds the 'engg4600' parent directory dynamically relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))              # .../compressible_2d
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # .../engg4600
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Analytical_Sod_Shock_Solver import sod_analytical_solver

plt.style.use(['science', 'no-latex'])   # no-latex: much faster for many frames

file_name_struc = 'Gif1Dd1q3_sod_Rho,U,Theta,Entropy_P00_*.pvti'
pvti_path = os.path.join(current_dir, 'TCLB_1D_Output_Vtis', 'Gif', file_name_struc)
gif_path  = os.path.join(current_dir, 'Presentation_Shock_Tube.gif')


def iteration_of(f):
    return int(re.search(r"_(\d+)\.pvti$", f).group(1))


files = sorted(glob.glob(pvti_path), key=iteration_of)   # numeric, not lexicographic
if not files:
    raise FileNotFoundError(f"No files match {pvti_path}")


#Physical Constants
gamma_g = 1.4                          #Heat-Capacity ratio
rg_phys = 287.0                        #Specific gas constant for air [J/Kg/K]
T_r     = 1460.0                       #Reference Temperature for nondimensionalisation of dt [K]
T_inf   = 300.0                        #Free-stream temperature [K]
p_inf   = 101325.0                     #Free-stream pressure [Pa]
rho_inf = p_inf / (rg_phys * T_inf)    #Free-stream density [kg/m3]

#Domain Discretisation
L_phys  = 1.0                                     #Channel Domain Length [m]
NX      = 401                                     #Nodes
dx_phys = L_phys / (NX - 1)                       #Delta X in Physical Units [m]
dt_phys = dx_phys / np.sqrt(3.0 * rg_phys * T_r)  #Delta T in Physical Units [s]
Cu      = dx_phys / dt_phys                       #Velocity unit [m/s]

NSTEPS    = 358         #Time Steps For Simulation
PLOT_XLIM = (0.0, L_phys)   # [m]
stride    = 8

#Lattice / Initial Conditions for Sod Shock tube
rho_L, rho_R = 3.0, 1.0      #Lattice Density Left and Right (must match the TCLB setup)
x_diaph = 0.5 * L_phys       #Diaphragm position [m] (must match the TCLB setup)

#Node positions [m], shared by TCLB and the analytic solution
x_phys = np.arange(NX) * dx_phys


def frame_data(f):
    #Lattice -> SI
    mesh = pv.read(f)
    rho = mesh["Rho"] / rho_R * rho_inf     # [kg/m3]  lattice rho_R  <->  rho_inf
    u   = mesh["U"][:, 0] * Cu              # [m/s]
    T   = mesh["Theta"] * T_r               # [K]      theta = T / T_r
    return rho, u, T


def analytic(time):
    #Returns rho [kg/m3], T [K], u [m/s]
    rho_left = rho_L / rho_R * rho_inf
    p_left   = rho_L / rho_R * p_inf
    if time == 0.0:
        #Initial step, solver is singular at t = 0
        rho = np.where(x_phys < x_diaph, rho_left, rho_inf)
        return rho, np.full(NX, T_inf), np.zeros(NX)
    rho_a, u_a, p_a = sod_analytical_solver(rho_left, p_left, 0.0,
                                            rho_inf, p_inf, 0.0, gamma_g,
                                            x_phys, x_diaph, time)
    T_a = p_a / (rho_a * rg_phys)
    return rho_a, T_a, u_a


#Pass 1: load all frames and analytic solutions
data = []
for f in files:
    it   = iteration_of(f)
    time = it * dt_phys
    data.append((it, time, frame_data(f), analytic(time)))


#Global y-limits over every frame (TCLB + analytic), with 5 % padding
def global_lim(idx_tclb, idx_an):
    lo = min(min(d[2][idx_tclb].min(), d[3][idx_an].min()) for d in data)
    hi = max(max(d[2][idx_tclb].max(), d[3][idx_an].max()) for d in data)
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad

# frame_data -> (rho, u, T);  analytic -> (rho, T, u)
RHO_LIM = global_lim(0, 0)
T_LIM   = global_lim(2, 1)
U_LIM   = global_lim(1, 2)


#Pass 2: plot
frames = []
for it, time, (rho_TCLB, u_TCLB, T_TCLB), (rho_an, T_an, u_an) in data:
    print(f"Gif is {it / NSTEPS * 100:.2f} % complete.")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

    #Analytic
    ax1.plot(x_phys, rho_an, "k-", lw=1.4, label="Analytic")
    #TCLB solver
    ax1.plot(x_phys[::stride], rho_TCLB[::stride], "o", color="tab:green", mfc="none", ms=5, lw=0, label="TCLB (HLBM)")
    ax1.plot(x_phys, rho_TCLB, ":", color="tab:green", lw=0.9)
    ax1.set_ylabel(r"$\rho$ [kg/m$^3$]")
    ax1.set_ylim(RHO_LIM)
    ax1.grid(alpha=0.25)

    #Analytic
    ax2.plot(x_phys, T_an, "k-", lw=1.4, label="Analytic")
    #TCLB solver
    ax2.plot(x_phys[::stride], T_TCLB[::stride], "o", color="tab:green", mfc="none", ms=5, lw=0, label="TCLB (HLBM)")
    ax2.plot(x_phys, T_TCLB, ":", color="tab:green", lw=0.9)
    ax2.set_ylabel(r"$T$ [K]")
    ax2.set_ylim(T_LIM)
    ax2.grid(alpha=0.25)

    #Analytic
    ax3.plot(x_phys, u_an, "k-", lw=1.4, label="Analytic")
    #TCLB solver
    ax3.plot(x_phys[::stride], u_TCLB[::stride], "o", color="tab:green", mfc="none", ms=5, lw=0, label="TCLB (HLBM)")
    ax3.plot(x_phys, u_TCLB, ":", color="tab:green", lw=0.9)
    ax3.set_ylabel(r"$u$ [m/s]")
    ax3.set_ylim(U_LIM)
    ax3.grid(alpha=0.25)

    ax1.legend(loc="lower left", fontsize=9)
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel(r"$x$ [m]")
    ax3.set_xlim(PLOT_XLIM)   # sharex applies it to all three
    fig.suptitle(rf"$t = {time * 1e3:.4f}$ ms, iteration {it}")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")   # no bbox_inches="tight", keeps frame size constant
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB"))

frames[0].save(gif_path, save_all=True, append_images=frames[1:],
               duration=40, loop=0, optimize=True)   # 100 ms = 10 fps
print(f"Saved {gif_path}")