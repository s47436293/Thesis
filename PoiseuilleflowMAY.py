# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:33:44 2026

@author: krekj
"""

import numpy as np
import matplotlib.pyplot as plt
import time

#Simulation Parameters
scale  = 3          # increase for finer grid (NX,NY scale linearly)
NX     = 5  * scale    # channel length (x, streamwise)
NY     = 5  * scale    # channel width  (y, wall-normal)
NSTEPS = int(1e4 * scale**2)
tau    = 0.9   # BGK relaxation — exact Poiseuille solution
omega  = 1.0 / tau
u_max  = 0.1 / scale           # maximum (centreline) velocity
nu     = (2*tau - 1) / 6       # kinematic viscosity
Re     = NY * u_max / nu       # Reynolds number

# Convergence detection
tol   = 1e-12    # relative change tolerance
teval = 100      # check every teval steps


#Lattice Definitions 
NPOP = 9
cx = np.array([ 1,  0, -1,  0,  1, -1, -1,  1,  0], dtype=int)
cy = np.array([ 0,  1,  0, -1,  1,  1, -1, -1,  0], dtype=int)
w  = np.array([1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36, 4/9])

#X Velocity Index Directions
U_pos = np.array([0,4,7])
U_neg = np.array([2,5,6])

#Y Velocity Index Directions 
V_pos = np.array([1, 4, 5])
V_neg = np.array([3,6,7])

#Bounce Back No slip Pairs

# Bounce-back index pairs
TOP_IN  = np.array([1, 4, 5])
TOP_OUT = np.array([3, 6, 7])
BOT_IN  = np.array([3, 6, 7])
BOT_OUT = np.array([1, 4, 5])

gradP      = 8 * nu * u_max / NY**2
rho_outlet = 1.0
rho_inlet  = 3*(NX - 1)*gradP + rho_outlet

#
x = np.arange(NX) + 0.5
y = np.arange(NY) + 0.5

u_analy = -4*u_max / NY**2 * (y - 0) * (y - NY)



def rho_calc(f):
    rho = np.sum(f,axis=-1)
    return rho

def vel_calc(f):
    u = np.sum(f[:,:,U_pos], axis = -1) - np.sum(f[:,:,U_neg], axis=-1)
    v = np.sum(f[:,:,V_pos], axis = -1) - np.sum(f[:,:,V_neg], axis=-1)    
    return u,v

def feq_calc(rho,u,v):
    feq = w * (rho[..., np.newaxis]
               + 3*(u[..., np.newaxis]*cx + v[..., np.newaxis]*cy))
    return feq


def f_stream(f):
    out = np.empty_like(f)
    for k in range(NPOP):
        out[:,:,k] = np.roll(np.roll(f[:,:,k], cx[k], axis=0), cy[k], axis=1)
    return out

def update(f):
    
    #Calculate macroscopic properties
    rho = rho_calc(f)
    u, v = vel_calc(f)
    

    #Equillibrium step
    feq = feq_calc(rho, u, v)
    
    #Collision step
    fstar = (1-omega)*f + omega*feq
    
    
    #Inlet PBBC ghost node at N=0
    u_N = u[NX-2]
    v_N = v[NX-2]
    rho_in = np.full(NY, rho_inlet)
    feq_in = feq_calc(rho_in, u_N, v_N)
    fstar[0,:, :] = feq_in + (fstar[NX-2, :, :] - feq[NX-2, :, :])
    
    #Outlet PBBC 
    u_0 = u[1]
    v_0 = v[1]
    rho_out = np.full(NY, rho_outlet)
    feq_out = feq_calc(rho_out, u_0, v_0)
    fstar[-1,:, :] = feq_out + (fstar[1, :, :] - feq[1, :, :])
    
    #Streaming step
    fprop = f_stream(fstar)
    
    #Bounce Back Boundary Conditions 
    
    #Top
    fprop[:, NY-1, TOP_OUT] = fstar[:, NY-1, TOP_IN]
    
    #Bottom
    fprop[:, 0, BOT_OUT] = fstar[:, 0, BOT_IN]
    
    
    return fprop, u 


def main():
    
    #Initialise density distrbution function:
    f     = np.zeros((NX, NY, NPOP))
    for k in range(NPOP):
        f[:, :, k] = w[k]
    fprop = f.copy()
        
    u_old = np.zeros((NX,NY))
    
    #Time iteration 
    for step in range(NSTEPS):
        fprop, u = update(fprop)
        
        #Convergence Step
        if (step+1) % teval == 0:
            conv = abs(u.mean() / (u_old.mean()) - 1)
            if conv < tol:
                break
            u_old = u.copy()
    
    
    #Post processing 
    rho_f = rho_calc(fprop)
    u_f, v_f = vel_calc(fprop)
    
    
    
    plt.figure(figsize=(6, 5))
    plt.plot(u_f[NX//2, :], y, "o-", color="steelblue", label="LBM")
    plt.plot(u_analy,        y, "-",  color="crimson",   lw=2, label="Analytical")
    plt.title("Poiseuille flow simulation with tau = 0.9")
    plt.xlabel("$u_x$");  plt.ylabel("$y$")
    plt.legend();  plt.grid(alpha=0.3);  plt.tight_layout()
    plt.show()   # FIX 3: correct syntax
        
if __name__ == "__main__":
    main()