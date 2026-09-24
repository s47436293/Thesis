# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:16:44 2026

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
import pandas as pd
import pyvista as pv

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

NSTEPS = 358 #Time Steps For Simulation
teval     = 50 # Print interval
PLOT_XLIM = (0.0, 1.0) #


#Lattice Constants (Dimensionless)
cs2 = 1.0 / 3.0 # D1Q3 sound speed square
rg  = cs2 # Lattice gas constant  (= cs2)
cv  = rg / (gamma_g - 1.0) # Specific heat, const. volume      (4.2.38)
cp  = gamma_g * rg / (gamma_g - 1.0)# Specific heat, const. presure (4.2.38)
dx  = 1.0
dt  = 1.0
x_latt = np.arange(NX, dtype=float)
theta_inf = T_inf / T_r #Dimensionless Temperature                    
mu_latt   = mu_phys * dt_phys / (rho_inf * dx_phys**2) #Dimensionless Viscosity
c_R_phys = np.sqrt(gamma_g * rg_phys * T_inf)


#Initial Conditions for Sod Shock tube

rho_L, rho_R     = 3.0, 1.0 #Density Left and Right
theta_L, theta_R = theta_inf, theta_inf #Temperature Left and Right
x_latt = np.arange(NX, dtype=float)
x_norm = x_latt / (NX - 1)


plt.style.use('science')
x_norm = x_latt / (NX - 1)

#Analytical Solution
rho_a, u_a, p_a = sod_analytical_solver(3.0*rho_inf, 3.0*p_inf, 0.0, rho_inf, p_inf, 0.0, gamma_g, x_norm*L_phys, 0.5*L_phys, NSTEPS*dt_phys) 

#Normalise Analytical For plot
T_a = p_a / (rho_a * rg_phys)
rho_norm_an = rho_a / rho_inf
T_norm_an   = T_a / T_inf
u_norm_an   = u_a / c_R_phys



#Importing Digitised Curves

# Base folder where  CSVs exist
csv_folder = os.path.join(project_root, 'ThesisCode', 'Compressible_1D', 'Florian_Digitised_SodShiock_CSVs')

# 1. Rho import
rho_csv_path = os.path.join(csv_folder, 'Florian_Digitised_rhoRhor.csv')
rho_norm_digi = pd.read_csv(rho_csv_path, header=None, names=["x", "y"])
rho_norm_digi = rho_norm_digi.sort_values("x")

# 2. T import
t_csv_path = os.path.join(csv_folder, 'Florian_Digitised_TTL.csv')
T_norm_digi = pd.read_csv(t_csv_path, header=None, names=["x", "y"])
T_norm_digi = T_norm_digi.sort_values("x")

# 3. U import 
u_csv_path = os.path.join(csv_folder, 'Florian_Digitised_uCr.csv')
U_norm_digi = pd.read_csv(u_csv_path, header=None, names=["x", "y"])
U_norm_digi = U_norm_digi.sort_values("x")


#Import 1D TCLB CURVES

x_norm_TCLB = (np.arange(NX) + 0.5) * dx_phys / L_phys
#Base Folder Where 1D Vtis exist
vti_Folder_1D = os.path.join(project_root, 'ThesisCode','Compressible_1D','TCLB_1D_Output_Vtis')
Sod_1D_TCLB_path = os.path.join(vti_Folder_1D, 'FlorianSOD1Dd1q3_sod_Rho,U,Theta,Entropy_P00_00000358.vti')

SOD_TCLB_1D_Data = pv.read(Sod_1D_TCLB_path)

rho_TCLB_1D = SOD_TCLB_1D_Data["Rho"]
u_TCLB_1D = SOD_TCLB_1D_Data["U"]
theta_TCLB_1D = SOD_TCLB_1D_Data["Theta"]

#Normalise Data

#Normalise TCLB Curves
rho_norm_TCLB_1D = rho_TCLB_1D/ rho_R
theta_norm_TCLB_1D = theta_TCLB_1D/theta_inf
u_norm_TCLB_1D   = u_TCLB_1D[:, 0] * Cu / c_R_phys


#Import 2D TCLB CURVES
vti_Folder_2D = os.path.join(project_root, 'ThesisCode','Compressible_2D','TCLB_2D_Output_Vtis')
Sod_2D_TCLB_path = os.path.join(vti_Folder_2D, 'FlorianSOD2Dd2q9_sod_VTK_P00_00000358.vti')

SOD_TCLB_2D_Data = pv.read(Sod_2D_TCLB_path)
nx, ny, _ = np.array(SOD_TCLB_2D_Data.dimensions) - 1     # points -> cells: 401, 4, 1
j  = ny // 2                                  # any row
sl = slice(j*nx, (j+1)*nx)                    # that row, x fastest

rho_TCLB_2D   = SOD_TCLB_2D_Data["Rho"][sl]
u_TCLB_2D     = SOD_TCLB_2D_Data["U"][sl]                    # still (nx, 3)
theta_TCLB_2D = SOD_TCLB_2D_Data["Theta"][sl]

#Normalise Data
rho_norm_TCLB_2D = rho_TCLB_2D/ rho_R
theta_norm_TCLB_2D = theta_TCLB_2D/theta_inf
u_norm_TCLB_2D   = u_TCLB_2D[:, 0] * Cu / c_R_phys



#Plot
with plt.style.context(["science","no-latex" ]):
    stride = 8
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.5, 8.5),sharex = True )
    
    #Analytic
    ax1.plot(x_norm, rho_norm_an, "k-", lw= 1.4, label = "Analytic")

    #Digitised Data From Florian Thesis
    ax1.plot(rho_norm_digi["x"][::stride], rho_norm_digi["y"][::stride], "o", color ="tab:blue", mfc="none", ms=5, lw=0, label="Florian Ref")
    ax1.plot(rho_norm_digi["x"], rho_norm_digi["y"], ":", color="tab:blue", lw=0.9)
    
    #1D TCLB Data
    ax1.plot(x_norm_TCLB[::stride], rho_norm_TCLB_1D[::stride], "o", color ="tab:green", mfc="none", ms=5, lw=0, label="1D TCLB")
    ax1.plot(x_norm_TCLB, rho_norm_TCLB_1D, ":", color="tab:green", lw=0.9) 
    
    #2D TCLB Data
    ax1.plot(x_norm_TCLB[::stride], rho_norm_TCLB_2D[::stride], "o", color ="tab:purple", mfc="none", ms=5, lw=0, label="2D TCLB")
    ax1.plot(x_norm_TCLB, rho_norm_TCLB_2D, ":", color="tab:purple", lw=0.9) 
    
    ax1.set_ylabel(r"$\rho\,/\,\rho_R$")
    ax1.grid(alpha=0.25)
    
    
    #Analytic
    ax2.plot(x_norm, T_norm_an, "k-", lw= 1.4, label = "Analytic")

    #Digitised Data From Florian Thesis
    ax2.plot(T_norm_digi["x"][::stride], T_norm_digi["y"][::stride], "o", color ="tab:blue", mfc="none", ms=5, lw=0, label="Florian Ref")
    ax2.plot(T_norm_digi["x"], T_norm_digi["y"], ":", color="tab:blue", lw=0.9)
    
    #1D TCLB Data
    ax2.plot(x_norm_TCLB[::stride], theta_norm_TCLB_1D[::stride], "o", color ="tab:green", mfc="none", ms=5, lw=0, label="1D TCLB")
    ax2.plot(x_norm_TCLB, theta_norm_TCLB_1D, ":", color="tab:green", lw=0.9) 

    #2D TCLB Data
    ax2.plot(x_norm_TCLB[::stride], theta_norm_TCLB_2D[::stride], "o", color ="tab:purple", mfc="none", ms=5, lw=0, label="2D TCLB")
    ax2.plot(x_norm_TCLB, theta_norm_TCLB_2D, ":", color="tab:purple", lw=0.9) 
    
    ax2.set_ylabel(r"$T\,/\,T_L$")
    ax2.grid(alpha=0.25)

    
    #Analytic
    ax3.plot(x_norm, u_norm_an, "k-", lw= 1.4, label = "Analytic")

    #Digitised Data From Florian Thesis
    ax3.plot(U_norm_digi["x"][::stride], U_norm_digi["y"][::stride], "o", color ="tab:blue", mfc="none", ms=5, lw=0, label="Florian Ref")
    ax3.plot(U_norm_digi["x"], U_norm_digi["y"], ":", color="tab:blue", lw=0.9)
    
    #1D TCLB Data
    ax3.plot(x_norm_TCLB[::stride], u_norm_TCLB_1D[::stride], "o", color ="tab:green", mfc="none", ms=5, lw=0, label="1D TCLB")
    ax3.plot(x_norm_TCLB, u_norm_TCLB_1D, ":", color="tab:green", lw=0.9) 
    
    #2D TCLB Data
    ax3.plot(x_norm_TCLB[::stride], u_norm_TCLB_2D[::stride], "o", color ="tab:purple", mfc="none", ms=5, lw=0, label="2D TCLB")
    ax3.plot(x_norm_TCLB, u_norm_TCLB_2D, ":", color="tab:purple", lw=0.9) 
    
    #TCLB solver
    ax3.set_ylabel(r"$u\,/\,c_R$")
    ax3.grid(alpha=0.25)
    
    ax1.legend(loc="lower left", fontsize=9)
    ax3.set_xlabel(r"$x$"); 
    ax3.set_xlim(PLOT_XLIM)
    #fig.suptitle("Hello", fontsize=9)
    fig.tight_layout()
    plt.show()
 

