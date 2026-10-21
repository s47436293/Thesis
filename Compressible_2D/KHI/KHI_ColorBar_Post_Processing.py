# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:56:06 2026

@author: krekj
"""

import os
import sys

# Finds the 'engg4600' parent directory dynamically relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__)) # KHI Folder

project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) # .../engg4600

if project_root not in sys.path:
    sys.path.insert(0, project_root)



import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pyvista as pv
import pandas as pd


plt.style.use('science')



#File paths for 128x128 grid
path_128_vtis = os.path.join(current_dir,'128x128','Output_Vtis')
file_name_struc_128 = 'McNallyKHI128KHI_VTK_P00_00000825.pvti'
PVTI_128 = os.path.join(path_128_vtis, file_name_struc_128)


#File paths for 256x256 grid
path_256_vtis = os.path.join(current_dir,'256x256','Output_Vtis')
file_name_struc_256 = 'McNallyKHI256KHI_VTK_P00_00001650.pvti'
PVTI_256 = os.path.join(path_256_vtis, file_name_struc_256)

#File paths for 512x512 grid
path_512_vtis = os.path.join(current_dir,'512x512','Output_Vtis')
file_name_struc_512 = 'McNallyKHI512KHI_VTK_P00_00003300.pvti'
PVTI_512 = os.path.join(path_512_vtis, file_name_struc_512)

fig_path = os.path.join(current_dir, 'Figures','Mcnally_KHI_Density_Resolution_t1p5')




def read(frame):
    mesh = pv.read(frame)
    # nx and ny are scalar integers
    nx, ny, _ = np.array(mesh.dimensions) - 1
    
    X, Y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny)
    rho = mesh['Rho'].reshape(ny, nx)
    
    # Removed tuple() from nx and ny
    return {"Rho":rho, "nx": nx, "ny": ny, "X": tuple(X), "Y": tuple(Y)}

Dict_128 = read(PVTI_128)
Dict_256 = read(PVTI_256)
Dict_512 = read(PVTI_512)

rhoMin = 0.9883; rhoMax =  2.0320
with plt.style.context("science"):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained")

    for a, (d, label) in zip(ax, [(Dict_128, "TCLB 128"),(Dict_256, "TCLB 256"),(Dict_512, "TCLB 512")]):
        im = a.imshow(d["Rho"], vmin=rhoMin, vmax=rhoMax, cmap="jet",origin="lower", extent=[0, 1, 0, 1])
        a.set_title(label)
        a.set_axis_off()

    fig.colorbar(im, ax=ax, ticks=np.arange(0.9, 2.01, 0.15), orientation="horizontal", location="bottom",shrink=0.6, aspect=40, label=r"$\rho$")
    plt.savefig(fig_path, dpi=300)
    plt.show()

