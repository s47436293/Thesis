# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:05:34 2026
Post Processing for Case 15 of 2D shock Problem explored in 
Bachelor Theis: Development of Compressible Lattice Boltzmann Methods.
Ai aided code for setting up plot similair to Florian Renard Thesis Figure.
Agent used: (Claude Opus 5)
@author: krekj
"""



import os
import sys

# Finds the 'engg4600' parent directory dynamically relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__)) # Case12 Folder
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) # .../engg4600

if project_root not in sys.path:
    sys.path.insert(0, project_root)



import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pyvista as pv

plt.style.use('science')

#Import Pvti

vti_path = os.path.join(current_dir,'Output_Vtis', 'ReportFigureSimulation')

Case_15_path = os.path.join(vti_path, 'Report_15d2q9_2D_RIEMANN_CASE_15_VTK_P00_00000310.pvti')
assert os.path.isfile(Case_15_path), Case_15_path
mesh = pv.read(Case_15_path)

#Figure Save path
fig_path = os.path.join(current_dir, 'figures','2D_Riemann_Case_15_Pressue_Contour')

#Simulation constants 
P_inf = 101325.0 #Pa
rho_inf = 1.176829 
#Plot components
nx, ny, _ = np.array(mesh.dimensions) - 1 
p = mesh.cell_data["Pressure"].reshape(ny,nx)[205:605, 205:605] / P_inf
rho = mesh.cell_data["Rho"].reshape(ny, nx)[205:605, 205:605] / rho_inf
U  = mesh.cell_data["U"].reshape(ny, nx, 3)[205:605, 205:605, :]
ux, uy = U[..., 0], U[..., 1]          # m/s, since U has unit="m/s"

s = 30  #arrow spacing 
vmin, vmax = float(np.nanmin(p)), float(np.nanmax(p))
span = (1.0 - 0.4) / 0.9          # 0.667
vmin = 0.4 - 0.05*span            # 0.367
vmax = vmin + span                # 1.033

x = (np.arange(400) + 0.5) / 400                   # cell centres, 0..1
X, Y = np.meshgrid(x, x)
with plt.style.context('science'):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    #vmin, vmax = np.nanmin(p), np.nanmax(p)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    #Plotting Pressure 
    im = ax.imshow(p, origin="lower", extent=[0, 1, 0, 1], cmap="jet", vmin= vmin, vmax=vmax, interpolation= "bilinear")
    ax.set_aspect("equal")
    #ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cb = fig.colorbar(im, cax=cax, ticks=np.arange(0.4, 1.001, 0.1), format="%.1f")
    cb.outline.set_linewidth(0.8)
    
    #Density Contours
    ax.contour(X, Y, rho,levels=np.arange(0.43, 0.9901, 0.02), colors="k", linewidths=0.4)
    
    #Velocity arrows
    ax.quiver(X[::s, ::s], Y[::s, ::s], ux[::s, ::s], uy[::s, ::s],color="k", pivot="mid", width=0.004)
    plt.savefig(fig_path, dpi=300)
    plt.show()




