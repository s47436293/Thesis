# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:07:53 2026

@author: James Krek
"""

import numpy as np
import matplotlib.pyplot as plt

#Physical Constraints of Simulation
rho_p = 1.0 #Density (kg/m^3)
nu_p = 0.01 #Kinematic viscosity (m^2/s) — FIXED, do not change
chan_length_p = 25.0  #Channel length (m)
chan_width_p = 20.0  #Channel width (m)
D_p = 1.0 #Cylinder diameter (m)

#Cylinder Location
cylinder_center_x = 10.0 / chan_length_p
cylinder_center_y = 10.0 / chan_width_p

Re = 100 #Reynolds Number of Sim
u_p = nu_p * Re / D_p #Fluid Velocity (m/s) Derived from Reynolds numeber and fluid viscosity 

#Chosen Simulation Parameters
taus = 0.55 #Relaxation tau
rho0s = 1.0 
deltaXs = 1.0 
deltaTs = 1.0 
deltaX = 0.1 #Lattice Spacing (M)

#Calculate deltaT using equation 7.14 LBM textbook
deltaT  = (1/3) * (taus - 0.5) * (deltaX**2 / nu_p)

#Conversion Factors
Cl = deltaX #Length Conversion Factor
Ct = deltaT #Time Conversion Factor
Cu = Cl / Ct #Velocity Conversion Factors
Crho = rho_p/rho0s

#Number of Nodes X and Y
NX  = int(chan_length_p / Cl)
NY  = int(chan_width_p  / Cl)
D_s = D_p / Cl #Cylinder Diamater (Lattice Units)


#Calculate fluid speed in lattice units and check stability of simulation
u_s = u_p / Cu # lattice velocity (lattice units)
u_max_est = 1.5 * u_s
print(u_s)
print(u_max_est)
if taus > 0.5 + (1/8 * u_s) and u_max_est < 0.4: #7.18 LBM textbook
    print("Stable")
else:
    print("Not Stable")

#Time step + other sim values
NSTEPS = 10000 #Number of Simulation Timesteps
omega  = 1.0 / taus 
teval  = 10
    

#D2Q9 Lattice Definitions
NPOP = 9
cx = np.array([ 1,  0, -1,  0,  1, -1, -1,  1,  0], dtype=int)
cy = np.array([ 0,  1,  0, -1,  1,  1, -1, -1,  0], dtype=int)
w  = np.array([1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36, 4/9])

#Directional Indices 
right_vel     = np.array([0, 4, 7])
left_vel      = np.array([2, 5, 6])
up_vel        = np.array([1, 4, 5])
down_vel      = np.array([3, 6, 7])
pure_vert_vel = np.array([1, 3, 8])
pure_horiz_vel = np.array([0, 2, 8])
lattice_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
oppo_indices  = np.array([2, 3, 0, 1, 6, 7, 4, 5, 8])

#Macroscopic Calculation Functions
def rho_calc(f):
    return np.sum(f, axis=-1)

def vel_calc(f, rho):
    u = (np.sum(f[:, :, right_vel], axis=-1) - np.sum(f[:, :, left_vel],  axis=-1)) / rho
    v = (np.sum(f[:, :, up_vel],   axis=-1) - np.sum(f[:, :, down_vel], axis=-1)) / rho
    return u, v

def feq_calc(rho, u, v):
    usqr = u**2 + v**2
    uc   = cx * u[..., np.newaxis] + cy * v[..., np.newaxis]
    return w * rho[..., np.newaxis] * (1 + 3*uc + 4.5*(uc**2) - 1.5*usqr[..., np.newaxis])

def f_stream(f):
    f_out = np.empty_like(f)
    for k in range(NPOP):
        f_out[:, :, k] = np.roll(
            np.roll(f[:, :, k], cx[k], axis=0),
                                  cy[k], axis=1
        )
    return f_out


#Geometry 
x = np.arange(NX)
y = np.arange(NY)
X, Y = np.meshgrid(x, y, indexing='ij')
cylinder_mask = (np.sqrt((X - NX*cylinder_center_x)**2 + (Y - NY*cylinder_center_y)**2)< (0.5 * D_s))

#Inlet Velocity Profile (Zou-He BC)
ux_profile = u_s * (1.0 + 1e-4 * np.sin(np.arange(NY) / (NY - 1) * 2 * np.pi)) #Small Asymmetry 


#NEBB (Zou-HE) Function

def nebb_left_wall(f, rho, u, v):
    f_wall = f[0,:,:] #(NY,9)
    
    #Known Populations 
    f_rest = f_wall[:,8]
    f_up = f_wall[: ,1]
    f_left = f_wall[:,2]
    f_down = f_wall[:,3]
    f_ul = f_wall[:, 5]
    f_dl = f_wall[:,6]
    
    uw_x = ux_profile #Velocity Profile on wall
    uw_y = 0.0 
    
    #Density at Wall
    rho_wall = (f_rest + f_up + f_down + 2.0 * (f_left + f_dl + f_ul)) / (1.0 - uw_x)
    
    f[0,:,0] = f_left + (2.0/3.0) * rho_wall * uw_x
    
    f[0,:,4] = f_ul - 0.5 * (f_up-f_down) + 0.5 * rho_wall * uw_y + (1.0/6.0) * rho_wall * uw_x
    
    f[0,:,7] = f_dl + 0.5 * (f_up-f_down) - 0.5 * rho_wall * uw_y + (1.0/6.0) * rho_wall * uw_x
    
    #Corrected Fields
    rho[0,:] = rho_wall
    u[0,:] = uw_x
    v[0,:] = uw_y

def zou_he_left_wall(f, rho, u, v):
    """
    Zou-He velocity inlet BC for left wall (x=0).
    
    Your D2Q9 convention:
      cx>0 unknown: 0=(+1,0), 4=(+1,+1), 7=(+1,-1)
      cx=0 known:   8=(0,0),  1=(0,+1),  3=(0,-1)
      cx<0 known:   2=(-1,0), 5=(-1,+1), 6=(-1,-1)
    
    Modifies f[0,:,:], rho[0,:], u[0,:], v[0,:] in-place.
    """
    f_wall = f[0, :, :]          # (Ny, 9)

    # Known populations (post-stream)
    f_rest = f_wall[:, 8]        # ( 0,  0)
    f_up   = f_wall[:, 1]        # ( 0, +1)
    f_left = f_wall[:, 2]        # (-1,  0)
    f_down = f_wall[:, 3]        # ( 0, -1)
    f_ul   = f_wall[:, 5]        # (-1, +1)
    f_dl   = f_wall[:, 6]        # (-1, -1)

    uw_x = ux_profile            # (Ny,)
    uw_y = 0.0

    # Density from known populations (eq. 5.49 / Zou-He)
    rho_wall = (f_rest + f_up + f_down + 2.0 * (f_left + f_ul + f_dl)) / (1.0 - uw_x)

    # Zou-He closure: f_i - f_i^eq = f_ī - f_ī^eq
    # Unknown right-moving populations set using their opposite known population
    # plus the difference of equilibria

    # f0 (+x, 0)  ←→  f2 (-x, 0)
    f[0, :, 0] = f_left + (2.0/3.0) * rho_wall * uw_x

    # f4 (+x,+y)  ←→  f6 (-x,-y)
    f[0, :, 4] = f_dl \
               + 0.5  * rho_wall * uw_y \
               + (1.0/6.0) * rho_wall * uw_x

    # f7 (+x,-y)  ←→  f5 (-x,+y)
    f[0, :, 7] = f_ul \
               - 0.5  * rho_wall * uw_y \
               + (1.0/6.0) * rho_wall * uw_x

    # Corrected macroscopic fields
    rho[0, :] = rho_wall
    u[0, :]   = uw_x
    v[0, :]   = uw_y  
    
#Update Function
def update(f_prev):
    #1. OutFlow BC Right Wall
    f_prev[-1,:, left_vel] = f_prev[-2,:, left_vel]
    
    #2. Calculate Macroscopic quantities
    rho = rho_calc(f_prev)
    u,v = vel_calc(f_prev,rho)
    
    #3. NEBB Dirchelt BC velocity inlet at left wall x==
    zou_he_left_wall(f_prev, rho, u, v)

    # 4. Equilibrium distribution
    f_eq = feq_calc(rho, u, v)

    # 5. BGK collision
    f_star = f_prev - omega * (f_prev - f_eq)

    # 6. Bounce-back on cylinder
    f_star[cylinder_mask, :] = f_prev[cylinder_mask][:, oppo_indices]

    # 7. Stream
    f_next = f_stream(f_star)

    return f_next

#initialise 
u_init = np.full((NX, NY), u_s)
v_init = np.zeros((NX, NY))
rho_init = np.ones((NX, NY))
f = feq_calc(rho_init, u_init, v_init)

#Prope location for fft
probe_x = int(NX * cylinder_center_x + 2.0 * D_s)
probe_y = int(NY * cylinder_center_y + 1.0 * D_s)
p_probe = np.zeros(NSTEPS)   #Pressure probe array

#Plot Setup
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.ion()

mesh_vel  = axes[0].pcolormesh((X*Cl).T, (Y*Cl).T, np.zeros((NX, NY)).T, cmap="inferno",
                                shading="auto", vmin=0*Cu, vmax=u_p * 1.5)
mesh_curl = axes[1].pcolormesh((X*Cl).T, (Y*Cl).T, np.zeros((NX, NY)).T, cmap="RdBu_r",
                                shading="auto", vmin=-0.03, vmax=0.03)

fig.colorbar(mesh_vel,  ax=axes[0], label="Velocity Magnitud")
fig.colorbar(mesh_curl, ax=axes[1], label="Vorticity")

cyl_x = (NX * cylinder_center_x)*Cl
cyl_y = (NY * cylinder_center_y)*Cl
for ax, title in zip(axes, ["Velocity Magnitude", "Vorticity (von Kármán Street)"]):
    ax.add_patch(plt.Circle((cyl_x, cyl_y), 0.5*D_p, color="white"))
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Metres)")
    ax.set_ylabel("y (Metres)")

title_text = fig.suptitle("Initialising...", fontsize=13)
fig.tight_layout()

for step in range(NSTEPS):
    f = update(f)
    rho_pt = rho_calc(f[probe_x, probe_y, :])
    p_probe[step] = rho_pt/ 3.0
    print(step)

    if step % teval == 0:
        rho_plot       = rho_calc(f)
        u_plot, v_plot = vel_calc(f, rho_plot)
        vel_mag        = np.sqrt(u_plot**2 + v_plot**2)*Cu
        if step> 1000:
            print(np.max(u_plot))
        du_dy = np.gradient(u_plot, axis=1)
        dv_dx = np.gradient(v_plot, axis=0)
        curl  = du_dy - dv_dx
        
        vel_mag[cylinder_mask] = np.nan
        curl[cylinder_mask]    = np.nan

        mesh_vel.set_array(vel_mag.T.ravel())
        mesh_curl.set_array(curl.T.ravel())

        t_phys = step * deltaT
        title_text.set_text(
            f"Step {step:,} / {NSTEPS:,}  |  t = {t_phys:.2f} s  "
            f"|  Re = {Re:.0f}  |  Ma = {u_s*np.sqrt(3):.3f}"
        )

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        #fig.savefig(f"vel.{step//teval:04d}.png", dpi=150, bbox_inches='tight')
        plt.pause(0.001)
        

plt.ioff()
from scipy.fft import fft, fftfreq

n_transient = int(50.0 * D_s / u_s) #Skip the values corresponding to first 50 times flow passes cylinder vortex shedding has not developed
signal = p_probe[n_transient:] - p_probe[n_transient:].mean()
N      = signal.size
freqs    = fftfreq(N, d=deltaTs)                   # frequencies in 1/timestep
spectrum = np.abs(fft(signal))

# Positive-frequency half, skip the DC bin
pos      = (freqs > 0)
freqs_p  = freqs[pos]
spec_p   = spectrum[pos]

peak_idx = np.argmax(spec_p)
f_shed_lu   = freqs_p[peak_idx]# per lattice timestep
f_shed_phys = f_shed_lu / Ct # Hz
St          = f_shed_lu * D_s / u_s

print("\n=== Vortex shedding analysis ===")
print(f"Probe (i, j)         : ({probe_x}, {probe_y})   "
      f"[2D downstream, 1D above centreline]")
print(f"Samples used         : {N}  (transient skipped: {n_transient} steps)")
print(f"f_shed (lattice)     : {f_shed_lu:.6f}  per timestep")
print(f"f_shed (physical)    : {f_shed_phys:.5f}  Hz")
print(f"Strouhal number St   : {St:.4f} ")

# Spectrum plot for sanity check
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.semilogy(freqs_p, spec_p, lw=1)
ax2.axvline(f_shed_lu, color='r', ls='--',
            label=f'f_peak = {f_shed_lu:.5f} /ts   (St = {St:.3f})')
ax2.set_xlim(0, 6 * f_shed_lu)
ax2.set_xlabel("Frequency (1 / lattice timestep)")
ax2.set_ylabel("|FFT(P_probe)|")
ax2.set_title("Spectrum of lattice pressure     at wake probe")
ax2.legend()
fig2.tight_layout()

# Time-series plot (post-transient) for visual confirmation of periodicity
fig3, ax3 = plt.subplots(figsize=(9, 4))
t_axis = (np.arange(N) + n_transient) * deltaT
ax3.plot(t_axis, signal, lw=0.8)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Pressure at probe (lattice units, mean-subtracted)")
ax3.set_title(f"Probe signal — period ≈ {1.0/f_shed_phys:.3f} s")
fig3.tight_layout()
# ==========================================================================

plt.show()
