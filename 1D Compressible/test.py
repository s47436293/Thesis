# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 21:52:26 2026

@author: krekj
"""
import numpy as np
import matplotlib.pyplot as plt
data  = np.loadtxt('Florian_Digitised_rho_rhoL.csv', delimiter=',', unpack = True)

x = data[:]
y = data[:,1]
x