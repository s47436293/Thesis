# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:41:45 2026

@author: krekj
"""
import os
import sys

# Finds the 'engg4600' parent directory dynamically relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))                  # KHI folder
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))    # .../engg4600
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pyvista as pv

# File paths for 512x512 grid
path_512_vtis = os.path.join(current_dir, '512x512', 'Output_Vtis')
file_name_struc_512 = 'McNallyKHI512KHI_VTK_P00_00003300.pvti'
PVTI_512 = os.path.join(path_512_vtis, file_name_struc_512)

fig_path = os.path.join(current_dir, 'Figures', 'Mcnally_KHI_Density_Resolution_512.png')


def read(frame):
    mesh = pv.read(frame)
    nx, ny, _ = np.array(mesh.dimensions) - 1
    rho = np.asarray(mesh['Rho']).reshape(ny, nx)
    return {"Rho": rho, "nx": nx, "ny": ny}


def plot_density(rho, fname, vmin, vmax):
    with plt.style.context("science"):
        plt.rcParams.update({"font.size": 9})

        fig = plt.figure(figsize=(2.6, 3.1))                   # width, height in inches
        ax  = fig.add_axes([0.02, 0.22, 0.96, 0.76])            # square plot area
        cax = fig.add_axes([0.06, 0.13, 0.88, 0.05])              # thin colorbar underneath

        im = ax.imshow(rho, origin="lower", extent=[0, 1, 0, 1],
                       cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_axis_off()

        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks(np.arange(1.05, 2.0, 0.15))
        cb.ax.minorticks_off()                                  # scienceplots adds these by default
        cb.ax.tick_params(which="both", labelsize=8, length=3, direction="out",
                          top=False, bottom=True)
        #cb.set_label("Density", fontsize=10, labelpad=1)
        cb.outline.set_linewidth(0.6)

        fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0.02)    
        plt.show()


Dict_512 = read(PVTI_512)

rhoMin, rhoMax = 0.9883, 2.0320
plot_density(Dict_512["Rho"], fig_path, rhoMin, rhoMax)