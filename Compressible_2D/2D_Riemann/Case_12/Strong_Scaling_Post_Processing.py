# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:07:53 2026

@author: krekj

"""
import os
import sys

# Finds the 'engg4600' parent directory dynamically relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))                  # KHI folder
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))    # .../engg4600
if project_root not in sys.path:
    sys.path.insert(0, project_root)


fig_path = os.path.join(current_dir, 'Figures', 'Strong_Scaling_Figure.png')
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
#Strong scaling
cores = [1.0,2.0,3.0,4.0 ]
f_


sim_time = np.array ([
    [112.004550, 112.811697, 113.044281],
    [66.874684, 65.807717, 67.575794],
    [53.646536, 54.988815, 56.820213],
    [53.931045, 54.649294, 64.319935],])
    
    
sim_time_average = np.mean(sim_time, axis=1)   
speedup = sim_time_average[0] / sim_time_average


with plt.style.context(["science"]):
    fig2, ax = plt.subplots( figsize = (5.0, 3.0))
    
    #Plotting Speedup
    ax.plot(cores, speedup, 'b--', label="Real Speedup")

    #Plot Reference
    ax.plot(cores, cores,'r-', label="Ideal Speedup")
    
    #Titles and Labels
    ax.set(xlabel=r"Number of CPU threads")
    ax.set(ylabel=r"Speedup")
    ax.grid()
    ax.legend()
    plt.savefig(fig_path, dpi=300)
    plt.show()
