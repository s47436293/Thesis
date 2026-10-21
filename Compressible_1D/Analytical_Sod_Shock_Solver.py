# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:59:30 2026

@author: krekj
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def sod_analytical_solver(rhoL, PL,uL, rhoR, PR, uR, gamma, x_grid, x0,t):
    
    #Compute WL and WR
    WL= np.array([rhoL, uL, PL])
    WR= np.array([rhoR, uR, PR])
    
    #Compute gamma related constants
    G1 = (gamma -1.0) / (2.0 * gamma)
    G2 = (gamma + 1.0)/ (2.0 * gamma)
    G3 = 2.0 * gamma / (gamma-1)
    G4 = 2.0 / (gamma - 1.0)
    G5 = 2.0 / (gamma + 1.0)
    G6 = (gamma - 1.0)/ (gamma + 1.0)
    G7 = (gamma - 1.0)/ 2.0
    G8 = gamma - 1.0
    
    
    #Calculate Sounds speeds in left and right states
    aL = np.sqrt(gamma * PL/ rhoL)
    aR = np.sqrt(gamma * PR/ rhoR)
    
    def fL(P):
        "Equation (4.6) in Toro"
        #Data dependant constants AL and BL
        AL = 2 / ((gamma +1) * rhoL)
        BL = (gamma -1 )/ (gamma + 1) * PL
        if  P<= PL: #Rarefraction Scenario
            return  (2 * aL / (gamma -1)) * ((P/PL)**G1 - 1)
        else: #Shock Scenario
            return (P - PL) * np.sqrt(AL / (P + BL))
            
    def fR(P):
        "Equation (4.7) in Toro"
        #Data dependant constants AR and BR
        AR = 2 / ((gamma + 1 ) * rhoR)
        BR = (gamma -1)/ (gamma + 1) * PR
        
        if P<= PR: #Rarefraction Scenario
            return (2 * aR / (gamma -1)) * ((P/PR)**G1-1)
        else: #Shock Scenario
            return (P - PR) * np.sqrt(AR / (P + BR))
    
    def f(P):
        """Equation (4.5) in Toro"""
        deltaU = uR - uL
        return fL(P) + fR(P) + deltaU
    
    #Initial Guess for P_Star and Root finding
    P_Star_Guess = 0.5 * (PL + PR)
    P_Star = fsolve(f,P_Star_Guess)[0]
    
    #Calculate UStar Equation (4.9) in toro
    u_Star = 0.5 * (uL + uR) + 0.5 * (fR(P_Star) - fL(P_Star)) 
    RightShock = True
    LeftShock = True
    if P_Star <= PL: #Left Rarefraction Wave
        rho_StarL = rhoL * (P_Star / PL)** (1/gamma) #Equation (4.53)
        a_StarL = aL * (P_Star / PL)** G1 #Equation (4.54)
        SHL = uL - aL #Equation (4.55)
        STL = u_Star - a_StarL #Equation (4.55)
        W_StarL = np.array([rho_StarL, u_Star, P_Star])
        LeftShock = False
    else: #Left Shock Wave
        rho_StarL = rhoL * ((P_Star/PL) + G6) / (G6* (P_Star/ PL) + 1) # Equation 4.50
        SL = uL - aL *np.sqrt( (G2 * (P_Star/ PL) + G1)) #Equation 4.52   
        W_StarL = np.array([rho_StarL, u_Star, P_Star])
        LeftShock = True
    if P_Star<= PR: # Right Rarefraction Wave
        rho_StarR = rhoR * (P_Star/ PR) ** (1/gamma)
        a_StarR = aR * (P_Star/ PR)** G1
        SHR = uR + aR
        STR = u_Star + a_StarR
        W_StarR = np.array([rho_StarR, u_Star, P_Star])
        RightShock = False
    else: #Right Shock Wave
        rho_StarR =  rhoR * ((P_Star/PR) + G6) / (G6* (P_Star/ PR) + 1) # Equation 4.57
        SR = uR + aR *np.sqrt( (G2 * (P_Star/ PR) + G1)) #Equation 4.59
        W_StarR = np.array([rho_StarR, u_Star, P_Star])
        RightShock =  True
        
    def WLfan(x,t):
        rhoN = rhoL * (G5 + G6/aL * (uL - (x/t)))**G4
        uN = G5 * (aL + G7 * uL + (x/t))
        PN = PL * (G5 + G6/aL * (uL - (x/t)))**G3  
        return np.array([rhoN, uN, PN])
    
    def WRfan(x, t):
        rhoN = rhoR * (G5 - G6/aR * (uR - (x/t)))**G4
        uN = G5 * (-aR + G7 * uR + (x/t))
        PN = PR * (G5 -  G6/aR * (uR - (x/t)))**G3  
        return np.array([rhoN, uN, PN])
    
    # 4. Sample across the spatial domain
    x_grid = np.asarray(x_grid, dtype = float)
    rho_arr = np.empty_like(x_grid); 
    u_arr= np.empty_like(x_grid)
    P_arr = np.empty_like(x_grid)
        
    
    for i, xi in enumerate(x_grid):
        dx = xi - x0
        dxt = dx / t
        
        #Left Side
        if dxt <= u_Star:
            if LeftShock:
                if SL<= dxt and dxt<= u_Star:
                    rho_arr[i], u_arr[i], P_arr[i] = W_StarL
                else:
                    rho_arr[i], u_arr[i], P_arr[i] = WL
            else:
                if dxt<= SHL:
                    rho_arr[i], u_arr[i], P_arr[i] = WL
                    
                elif SHL<= dxt and dxt<= STL:
                    rho_arr[i], u_arr[i], P_arr[i] = WLfan(dx, t)
                    
                elif STL<= dxt and dxt<= u_Star:
                    rho_arr[i], u_arr[i], P_arr[i] = W_StarL
                    
                    
        elif u_Star <= dxt:
            if RightShock:
                
                if u_Star<= dxt and dxt<= SR:
                    rho_arr[i], u_arr[i], P_arr[i] = W_StarR
                else: 
                    rho_arr[i], u_arr[i], P_arr[i] = WR
            else: 
                if u_Star<= dxt and dxt<= STR:
                    rho_arr[i], u_arr[i], P_arr[i] = W_StarR
                    
                elif STR<= dxt and dxt<= SHR:
                    rho_arr[i], u_arr[i], P_arr[i] = WRfan(dx, t)     
                    
                else:
                    rho_arr[i], u_arr[i], P_arr[i] = WR                   
       
    return rho_arr, u_arr, P_arr

def main():
    Pinf = 101325
    rhoinf = Pinf/(300.0*287)
    x = np.linspace(0.0, 1.0, 1000)
    # --- RUN AND PLOT STANDARD SOD TUBE PROFILE ---
    rho, u, P = sod_analytical_solver(3.0*rhoinf, 3*Pinf, 0.0, rhoinf, Pinf, 0.0, 1.4, x, 0.5,7.984e-4)
    
    fig, axs = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    axs[0].plot(x, rho, 'r-', label='Density ($\rho$)')
    axs[0].set_ylabel('Density')
    axs[0].grid(True)
    
    axs[1].plot(x, P, 'b-', label='Pressure ($P$)')
    axs[1].set_ylabel('Pressure')
    axs[1].grid(True)
    
    axs[2].plot(x, u, 'g-', label='Velocity ($u$)')
    axs[2].set_ylabel('Velocity')
    axs[2].set_xlabel('Position ($x$)')
    axs[2].grid(True)
    
    plt.suptitle('Exact Analytical Solution to Sod Shock Tube ($t=0.2$)', fontsize=14)
    plt.tight_layout()
    plt.show()

                
if __name__ =="__main__":
    main()
                
        
        
                
            
            


