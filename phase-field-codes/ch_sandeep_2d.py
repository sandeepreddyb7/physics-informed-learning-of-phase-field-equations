#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:08:39 2022

@author: sandeep
"""

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
    phi = np.reshape(phi, (129, 129))
    phi_fx = np.roll(phi, 1, axis=0)    # forward in x
    phi_bx = np.roll(phi, -1, axis=0)   # backward in x
    phi_fy = np.roll(phi, 1, axis=1)    # forward in y
    phi_by = np.roll(phi, -1, axis=1)   # backward in y
    phi_lap = (phi_fx + phi_bx + phi_fy + phi_by - 4 * phi) / (dx**2)
    
    gamma_1 = 2.5e-6      # Sort of interface energy, larger value, more diffuse, more round shapes
    mobility = 1.0 # Mobility of the interface
    gamma_2 = 1e-2
    f_phi = gamma_2*(np.power(phi,3)-phi)
    mu = f_phi - gamma_1*phi_lap
    
    mu_fx = np.roll(mu, 1, axis=0)    # forward in x
    mu_bx = np.roll(mu, -1, axis=0)   # backward in x
    mu_fy = np.roll(mu, 1, axis=1)    # forward in y
    mu_by = np.roll(mu, -1, axis=1)   # backward in y
    mu_lap = (mu_fx + mu_bx  + mu_fy + mu_by - 4 * mu) / (dx**2)
    f_y_value = mobility * ( mu_lap )
    
    f_y_value = np.reshape(f_y_value,-1)
    return f_y_value



def calc_mu(phi):
    dx = 1/64
    #dy = 1/512
    #phi = np.reshape(phi, (129, 129))
    phi_fx = np.roll(phi, 1, axis=0)    # forward in x
    phi_bx = np.roll(phi, -1, axis=0)   # backward in x
    phi_fy = np.roll(phi, 1, axis=1)    # forward in y
    phi_by = np.roll(phi, -1, axis=1)   # backward in y
    phi_lap = (phi_fx + phi_bx + phi_fy + phi_by - 4 * phi) / (dx**2)
    
    gamma_1 = 2.5e-6      # Sort of interface energy, larger value, more diffuse, more round shapes
    mobility = 1.0 # Mobility of the interface
    gamma_2 = 1e-2
    f_phi = gamma_2*(np.power(phi,3)-phi)
    mu = f_phi - gamma_1*phi_lap
    
    return mu

# Main run part
# Initialization / geometry and parameters
N = 128           # system size in x
Ny = 128            # system size in y
Nt = 200

end_t= 1
dt = end_t/Nt

x = np.linspace(-1,1,N+1)
y = np.linspace(-1, 1,Ny+1)
t = np.linspace(0,end_t,Nt+1)
t_eval_v = list(np.linspace(0,end_t,Nt+1))

X, Y= np.meshgrid(x,y,indexing='ij')

# Phi is the scalar field (order parameter), 0 means the first phase, 1 means the second phase

#Phi = (X**2)*np.sin(2*np.pi*X)
#Phi = (X**2)*np.cos(np.pi*X)

#Phi_ini = (x**2)*np.cos(np.pi*x)

#r = 0.3*centre
r = 0.4
centre = 0;
#R1 = np.sqrt((X-centre-r)**2+(Y-centre)**2)
#R2 = np.sqrt((X-centre+r)**2+(Y-centre)**2)
R1 = np.sqrt((X-centre-0.7*r)**2+(Y-centre)**2)
R2 = np.sqrt((X-centre+0.7*r)**2+(Y-centre)**2)
#epsilon = 3
epsilon =0.1
a1 = np.tanh((r-R1)/epsilon)
a2 = np.tanh((r-R2)/epsilon)
Phi = np.maximum(a1,a2)

mu = calc_mu(Phi)

Phi_r = np.reshape(Phi,-1)
#Phi_ini =-np.cos(2*np.pi*x)
#test = f_y(t,Phi_ini)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
h1 = ax.imshow(Phi, interpolation='nearest', 
        extent=[x.min(), x.max(), x.min(), x.max()], 
        origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h1, cax=cax)
cbar.ax.tick_params(labelsize=15) 
#ax.set_title('Frame = '+str(frame))
ax.set_title('$\phi$')

ax = fig.add_subplot(122)
h2 = ax.imshow(mu, interpolation='nearest', 
        extent=[x.min(), x.max(), x.min(), x.max()], 
        origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h2, cax=cax)
cbar.ax.tick_params(labelsize=15) 
ax.set_title('$\mu$')
fig.tight_layout(pad=3.0)




sol = solve_ivp(f_y,[0,end_t],Phi_r,method='Radau',t_eval = t_eval_v)

phi_out = sol.y

phi_out1 = np.reshape(phi_out,(129,129,Nt+1))

frame =0

for i in range(0, Nt+1):
    
    Phi = phi_out1[:,:,i]
    mu = calc_mu(Phi)
    frame = frame + 1

    # plotting and reporting every 10 steps, note that mean(Phi) is constant here, the system is conserved. Phi can be
    # solute concentration for example.
    if np.mod(frame, 10) == 0:
        print("Frame = ", frame, "/ Max Phi = ", np.max(Phi), "/ Min Phi =  ",
              np.min(Phi), "/ Mean Phi =", np.mean(Phi))
        # plotting part
        #plt.clf()
        #plt.imshow(Phi, interpolation='bilinear')
        #plt.colorbar()
        #plt.title("Frame = " + str(frame))
        #plt.draw()
        # plt.clim(0, 1)
        #plt.pause(0.00000001)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        h1 = ax.imshow(Phi, interpolation='nearest', 
                        extent=[x.min(), x.max(), x.min(), x.max()], 
                        origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h1, cax=cax)
        cbar.ax.tick_params(labelsize=15) 
        #ax.set_title('Frame = '+str(frame))
        ax.set_title('$\phi$')

        ax = fig.add_subplot(122)
        h2 = ax.imshow(mu, interpolation='nearest', 
                        extent=[x.min(), x.max(), x.min(), x.max()], 
                        origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h2, cax=cax)
        cbar.ax.tick_params(labelsize=15) 
        ax.set_title('$\mu$')
        fig.tight_layout(pad=3.0)
        plt.show()

# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111)
# h1 = ax.imshow(test1, interpolation='nearest', 
#         extent=[x.min(), x.max(), y.min(), y.max()], 
#         origin='lower', aspect='auto')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.10)
# cbar = fig.colorbar(h1, cax=cax)
# cbar.ax.tick_params(labelsize=15) 
# #ax.set_title('Frame = '+str(frame))
# ax.set_title('$\phi$')

    
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111)
# h1 = ax.imshow(sol.y, interpolation='nearest', 
#         extent=[t.min(), t.max(), x.min(), x.max()], 
#         origin='lower', aspect='auto')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.10)
# cbar = fig.colorbar(h1, cax=cax)
# cbar.ax.tick_params(labelsize=15) 
# #ax.set_title('Frame = '+str(frame))
# ax.set_title('$\phi$')

# Phi_all = {}
# Phi_all['u'] = sol.y
# Phi_all['x'] = X
# Phi_all['t'] = T

# np.save('ch_Phi_1e6_1e2.npy', Phi_all)


