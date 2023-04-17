#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:34:19 2022

@author: sandeep
"""

# Simple Allen-Cahn phase-field model
import numpy as np
from matplotlib import pyplot as plt
from sys import exit
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.integrate import solve_ivp





def f_y(t,phi):
    dx = 1/64
    #dy = 1/512
    phi_fx = np.roll(phi, 1, axis=0)    # forward in x
    phi_bx = np.roll(phi, -1, axis=0)   # backward in x
    #phi_fy = np.roll(phi, 1, axis=1)    # forward in y
    #phi_by = np.roll(phi, -1, axis=1)   # backward in y
    phi_lap = (phi_fx + phi_bx - 2 * phi) / (dx**2)
    
    gamma_1 = 1e-6      # Sort of interface energy, larger value, more diffuse, more round shapes
    mobility = 1.0 # Mobility of the interface
    gamma_2 = 1e-2
    f_phi = gamma_2*(np.power(phi,3)-phi)
    mu = f_phi - gamma_1*phi_lap
    
    mu_fx = np.roll(mu, 1, axis=0)    # forward in x
    mu_bx = np.roll(mu, -1, axis=0)   # backward in x
    #phi_fy = np.roll(phi, 1, axis=1)    # forward in y
    #phi_by = np.roll(phi, -1, axis=1)   # backward in y
    mu_lap = (mu_fx + mu_bx  - 2 * mu) / (dx**2)
    f_y_value = mobility * ( mu_lap )
    
    return f_y_value

    

# Main run part
# Initialization / geometry and parameters
N = 128           # system size in x
Ny = 2            # system size in y
Nt = 200

end_t= 1
dt = end_t/Nt

x = np.linspace(-1,1,N+1)
#y = np.linspace(-1/(2*N), 1/(2*N),Ny)
t = np.linspace(0,end_t,Nt+1)
t_eval_v = list(np.linspace(0,end_t,Nt+1))

X, T= np.meshgrid(x,t_eval_v,indexing='ij')

# Phi is the scalar field (order parameter), 0 means the first phase, 1 means the second phase

#Phi = (X**2)*np.sin(2*np.pi*X)
#Phi = (X**2)*np.cos(np.pi*X)

#Phi_ini = (x**2)*np.cos(np.pi*x)

Phi_ini =-np.cos(2*np.pi*x)
test = f_y(t,Phi_ini)

sol = solve_ivp(f_y,[0,1],Phi_ini, t_eval = t_eval_v)

# Some fun tests here
# phase 1 (Phi=0) is more stable at low temperature, phase 2 (Phi=1) is more stable at high temperatures.
# This is set in the form of the energy. At 500K, both phases have the same energy and evolution is controlled
# based on the initial random distribution and the interface energy


# frame = 0
# Phi_all = np.zeros((N,Nt+1))
# Phi_all[:,0] = Phi[:,0]
# for i in range(0, Nt):
#     Phi = update_phi(Phi,dt,N)
#     frame = frame + 1
#     Phi_all[:,i+1] = Phi[:,0]
#     # plotting and reporting part, note that mean(Phi) is not constant here, the system is non-conserve. Phi can be
#     # phase fraction, for example in a displasive phase transformation like fcc --> bcc
#     if np.mod(frame, 100) == 1:
#         print("Time = ", frame*dt,"Frame = ", frame, "/ Max Phi = ", np.max(Phi), "/ Min Phi =  ",
#               np.min(Phi), "/ Mean Phi =", np.mean(Phi))
    
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
h1 = ax.imshow(sol.y, interpolation='nearest', 
        extent=[t.min(), t.max(), x.min(), x.max()], 
        origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h1, cax=cax)
cbar.ax.tick_params(labelsize=15) 
#ax.set_title('Frame = '+str(frame))
ax.set_title('$\phi$')

Phi_all = {}
Phi_all['u'] = sol.y
Phi_all['x'] = x
Phi_all['t'] = t

np.save('data/ch_Phi_1e6_1e2.npy', Phi_all, allow_pickle=True)

phi_data = np.load('data/ch_Phi_1e6_1e2.npy', allow_pickle=True)
