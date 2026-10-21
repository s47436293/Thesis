# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 20:26:08 2026

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

path_vtis = os.path.join(current_dir, 'Output_Vtis')

#PVTI paths
PVTI_T2 = os.path.join(path_vtis, 'T2-8McNallyKHI512KHI_VTK_P00_00004435.pvti')
PVTI_T4 = os.path.join(path_vtis, 'T2-8McNallyKHI512KHI_VTK_P00_00008870.pvti')
PVTI_T6 = os.path.join(path_vtis, 'T2-8McNallyKHI512KHI_VTK_P00_00013305.pvti')
PVTI_T8 = os.path.join(path_vtis, 'T2-8McNallyKHI512KHI_VTK_P00_00017740.pvti')


mesh  = pv.read(PVTI_T2)

def read(frame):
    mesh = pv.read(frame)
    # nx and ny are scalar integers
    nx, ny, _ = np.array(mesh.dimensions) - 1
    
    X, Y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny)
    rho = mesh['Rho'].reshape(ny, nx)
    
    # Removed tuple() from nx and ny
    return {"Rho":rho, "nx": nx, "ny": ny, "X": tuple(X), "Y": tuple(Y)}

Dict_T2 = read(PVTI_T2)
Dict_T4 = read(PVTI_T4)
Dict_T6 = read(PVTI_T6)
Dict_T8 = read(PVTI_T8)
rhoMin = 0.9883; rhoMax =  2.0320
with plt.style.context("science"):

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    
    # Added the 4th dictionary tuple here
    data_list = [(Dict_T2, 'T=2'),(Dict_T4, 'T=4'),(Dict_T6, 'T=6'), (Dict_T8, 'T=8')  ]
    
    # ax.flat seamlessly loops through all 4 subplots
    for a, (d, label) in zip(ax.flat, data_list):
        im = a.imshow(d['Rho'], vmin=rhoMin, vmax=rhoMax, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
        a.set_title(label)
        a.set_axis_off()
    
    # The colorbar will now balance perfectly beneath the full 2x2 grid
    fig.colorbar(im, ax=ax, ticks=np.arange(0.9, 2.01, 0.15), orientation='horizontal', location='bottom', shrink=0.6, aspect=40, label=r'$\rho$')
    
    #plt.savefig(fig_path, dpi=300)
    plt.show()
