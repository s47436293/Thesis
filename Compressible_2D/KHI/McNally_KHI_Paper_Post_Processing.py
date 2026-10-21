# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 12:55:08 2026

@author: krekj
"""

import os
import sys
import glob, re

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

#File Path

fig_path1 = os.path.join(current_dir, 'Figures', 'Mcnally_Mode_Growth')
fig_path2 =  os.path.join(current_dir, 'Figures', 'Mcnally_Y_KE')

#File paths for reference data
ref_path = os.path.join(current_dir,'McNallyReferenceData')
ref_amp_path = os.path.join(ref_path, 'khmode_rev0.txt')
ref_kinener_path = os.path.join(ref_path, 'khener_rev1.txt')

#importing mode data
ref_mode = pd.read_csv(ref_amp_path, sep=r"\s+", comment="#", header=None,names=["time", "amplitude", "uncertainty"])

#importing Kinetic energy data
ref_kinener_path = os.path.join(ref_path, 'khener_rev1.txt')
ref_ener = pd.read_csv(ref_kinener_path, sep=r"\s+", comment="#", header=None,names=["time", "maxKE"])



#File paths for 128x128 grid
path_128_vtis = os.path.join(current_dir,'128x128','Output_Vtis')
file_name_struc_128 = 'McNallyKHI128KHI_VTK_P00*.pvti'
PVTIs_128 = os.path.join(path_128_vtis, file_name_struc_128)
files_128 = sorted(glob.glob(PVTIs_128))

#File paths for 256x256 grid
path_256_vtis = os.path.join(current_dir,'256x256','Output_Vtis')
file_name_struc_256 = 'McNallyKHI256KHI_VTK_P00*.pvti'
PVTIs_256 = os.path.join(path_256_vtis, file_name_struc_256)
files_256 = sorted(glob.glob(PVTIs_256))

#File paths for 512x512 grid
path_512_vtis = os.path.join(current_dir,'512x512','Output_Vtis')
file_name_struc_512 = 'McNallyKHI512KHI_VTK_P00*.pvti'
PVTIs_512 = os.path.join(path_512_vtis, file_name_struc_512)
files_512 = sorted(glob.glob(PVTIs_512))


UG = np.sqrt(3 * 1.0 * 6.25)

def diagnostic_values(path):

    m = pv.read(path)
    nx, ny, _ = np.array(m.dimensions) -1

    # VTK cell data is ordered x-fastest -> (ny, nx, 3); component 1 is v_y
    Vy = m.cell_data["U"].reshape(ny, nx, 3)[:, :, 1]
    rho = m.cell_data["Rho"].reshape(ny, nx)

    # cell centres of the unit periodic box
    X, Y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny)

    # weight peaks at the two shear layers, y = 0.25 and y = 0.75
    di = np.exp(-4*np.pi*np.where(Y < 0.5,np.abs(Y - 0.25),np.abs((1 - Y) - 0.25)))

    si = Vy * np.sin(4*np.pi*X) * di
    ci = Vy * np.cos(4*np.pi*X) * di

    D = di.sum()
    M = 2.0*np.sqrt((si.sum()/D)**2 + (ci.sum()/D)**2)
    KE = 0.5 * Vy**2 * rho
    return [float(M), float(np.max(KE))]


def snapshot_times(files, N):

    it = np.array([int(re.search(r"_(\d+)\.pvti$", f).group(1)) for f in files])
    return it * (1.0 / N) / UG



#Calculate time arrays for each grid size
t_128= snapshot_times(files_128, 128)
t_256 = snapshot_times(files_256, 256)
t_512 = snapshot_times(files_512,512)



#Calculate Mode amplitude and Kinetic Energy for each grid size over time
ModeAmp_128 = []
KE_128 = []
for f in files_128:
    val_128 = diagnostic_values(f)
    ModeAmp_128.append(val_128[0])
    KE_128.append(val_128[1])


ModeAmp_256 = []
KE_256 = []
for f in files_256:
    val = diagnostic_values(f)
    ModeAmp_256.append(val[0])
    KE_256.append(val[1])


ModeAmp_512= []
KE_512 = []
for f in files_512:
    val = diagnostic_values(f)
    ModeAmp_512.append(val[0])
    KE_512.append(val[1])

#Calcualte amplitude growth rate
dm_dt_128 = np.gradient(ModeAmp_128, t_128)
dm_dt_256 = np.gradient(ModeAmp_256, t_256)
dm_dt_512 = np.gradient(ModeAmp_512, t_512)

dm_dt_ref = np.gradient((ref_mode['amplitude']),ref_mode['time'])

#Create Mode Growth Plot
with plt.style.context(["science"]):
    fig1, (ax1,ax2) = plt.subplots(1,2, figsize = (8.0, 3.0))
    #Plotting Amplitude
    
    
    #Plotting TCLB Curves
    ax1.semilogy(t_128, ModeAmp_128, 'r--', label = "TCLB128")
    ax1.semilogy(t_256, ModeAmp_256, 'g--', label = "TCLB256")
    ax1.semilogy(t_512, ModeAmp_512, 'c--', label = "TCLB512")

    #Plotting Reference
    ax1.semilogy(ref_mode['time'], ref_mode['amplitude'], 'b-', label = "Reference")


    #Titles and Labels
    ax1.set(xlabel=r"Time ")
    ax1.set(ylabel=r"Amplitude")
    ax1.grid()
    ax1.legend()
    
    
    #Plottign Amplitude Growth rate
    #Plotting TCLB Curves
    ax2.plot(t_128, dm_dt_128, 'r--', label = "TCLB128")
    ax2.plot(t_256, dm_dt_256, 'g--', label = "TCLB256")
    ax2.plot(t_512, dm_dt_512, 'c--', label = "TCLB512")
    
    #Plotting Reference
    ax2.plot(ref_mode['time'], dm_dt_ref, 'b-', label = "Reference")
    
    #Titles and Labels
    ax2.set(xlabel=r"Time ")
    ax2.set(ylabel=r"Amplitude Growth Rate")
    ax2.grid()
    plt.savefig(fig_path1, dpi=300)
    ax2.legend()
    
    #plt.show()
    
#Creat Maximum y-direction kinetic energy plot
with plt.style.context(["science"]):
    fig2, ax = plt.subplots( figsize = (4.0, 1.5))
    
    #Plotting TCLB Curves
    ax.semilogy(t_128, KE_128, 'r--', label = "TCLB128")
    ax.semilogy(t_256, KE_256, 'g--', label = "TCLB256")
    ax.semilogy(t_512, KE_512, 'c--', label = "TCLB512")
    
    #Plot Reference
    ax.semilogy(ref_ener['time'],ref_ener['maxKE'],'b-', label='Reference')
    
    #Titles and Labels
    ax.set(xlabel=r"Time ")
    ax.set(ylabel=r"$\max(1/2 \rho v_y^2)$")
    ax.grid()
    ax.legend()
    plt.savefig(fig_path2, dpi=300)
    plt.show()
    
    
    


#Figure Save path
