# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:35:55 2026

@author: krekj
"""

import os 
import sys
import glob, re

# Finds the 'engg4600' parent directory dynamically relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__)) # Case12 Folder
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) # .../engg4600

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Global imports now work seamlessly anywhere

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import scienceplots
import pyvista as pv

plt.style.use('science')
file_name_struc = 'Gif_12d2q9_2D_RIEMANN_CASE_12_VTK_P00_*.pvti'
pvti_path = os.path.join(current_dir,'Output_Vtis', 'GifFigureSimulation',file_name_struc)


gif_path = os.path.join(current_dir, 'figures','Case_12.gif')
files = sorted(glob.glob(pvti_path))

#Simulation Parameters
del_t = 2.1998e-6; sim_time = 387 * del_t

def frame_data(f):
    m = pv.read(f)
    nx, ny, _ = np.array(m.dimensions) - 1
    p   = m.cell_data["Pressure"].reshape(ny, nx)[205:605, 205:605]/ 101325.0
    rho = m.cell_data["Rho"].reshape(ny, nx)[205:605, 205:605] / 1.176829
    return p, rho


P0= frame_data(files[0])[0]
rho0= frame_data(files[0])[1]

vmin = 0.3264819529324922
vmax = 2.5267295714754914
x = (np.arange(400) + 0.5) / 400                   # cell centres, 0..1
X, Y = np.meshgrid(x, x)

import io, os
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
frames = []
for f in files:
    p, rho = frame_data(f)
    
    it = int(re.search(r"_(\d+)\.pvti$", f).group(1))
    time = it * del_t
    fin_pct = time/ sim_time * 100
    print(f"Gif is {fin_pct:.2f} % complete.")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    im = ax.imshow(p, origin="lower", extent=[0, 1, 0, 1], cmap="jet", vmin=vmin, vmax=vmax)
    ax.contour(X, Y, rho, levels=np.arange(0.54, 1.70001, 0.04), colors="k", linewidths=0.4)
    ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f" t ={time:.4f}, iteration {it}")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im, cax=cax, ticks=np.arange(0.50, 2.2501, 0.25), format="%.2f")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")          # no bbox_inches="tight"
    plt.close(fig)
    frames.append(Image.open(buf))

frames[0].save(gif_path, save_all=True, append_images=frames[1:],
               duration=100, loop=0, optimize=True)      # 100 ms = 10 fps











"""


with plt.style.context('science'):
    fig, ax = plt.subplots(figsize=(5, 5))
    #Plotting Pressure 
    im = ax.imshow(P0, origin="lower", extent=[0, 1, 0, 1], cmap="jet", vmin= vmin, vmax = vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    fig.colorbar(im, label=r"$\rho / \rho_\infty$")
    #Density Contours
    ax.contour(X, Y, rho0, levels=np.arange(0.54, 1.7, 0.04), colors="k", linewidths=0.4)
    plt.show()


LEVELS = np.arange(0.54, 1.70001, 0.04)
cs = None

def update(i):
    global cs
    p, rho = frame_data(files[i])
    im.set_data(p)
    if cs is not None:
        try:
            cs.remove()                      # matplotlib >= 3.8
        except AttributeError:
            for c in cs.collections:         # older versions
                c.remove()
    cs = ax.contour(X, Y, rho, levels=LEVELS, colors="k", linewidths=0.4)
    return [im]

anim = FuncAnimation(fig, update, frames=len(files), blit=False)

anim.save(gif_path, writer=PillowWriter(fps=10))
"""
