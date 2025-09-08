# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:22:01 2024

@author: ektop
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# 1. **Polarization Squeezing**

def plot_polarization_squeezing():
    r_values = np.linspace(0, 2, 100)
    Delta_x_squared = 0.5 * np.exp(-2 * r_values)
    Delta_y_squared = 0.5 * (1 + np.random.rand(len(r_values)))  # Assuming some random uncertainty for Delta_y

    Delta_squared = Delta_x_squared + Delta_y_squared

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, Delta_squared, label='Combined Variance $\Delta^2$', color='blue')
    plt.plot(r_values, 0.5 * np.ones(len(r_values)), 'r--', label='Vacuum State Limit (0.5)')
    plt.ylim(0, 1)
    plt.xlabel('Squeezing Parameter $r$', fontsize=14)
    plt.ylabel('Variance', fontsize=14)
    plt.title('Polarization Squeezing', fontsize=16)
    plt.legend()
    plt.grid()
    plt.show()

# 2. **Fibered Interferometry**

def plot_fibered_interferometry():
    k = 2 * np.pi / 810e-9  # Wavenumber for 532 nm light
    L_eff = np.linspace(0, 20, 100)  # Effective arm length in meters
    h_t = np.linspace(0, 1e-22, 100)  # Strain from gravitational waves

    L_eff_mesh, h_t_mesh = np.meshgrid(L_eff, h_t)

    Delta_phi = k * L_eff_mesh * h_t_mesh

    plt.figure(figsize=(10, 6))
    plt.contourf(L_eff, h_t, Delta_phi, levels=50, cmap='viridis')
    plt.colorbar(label='Differential Phase Shift $\Delta \phi$ (rad)')
    plt.xlabel('Effective Arm Length $L_{eff}$ (m)', fontsize=14)
    plt.ylabel('Strain $h(t)$ (unitless)', fontsize=14)
    plt.title('Differential Phase Shift in Fibered Interferometry', fontsize=16)
    plt.show()


# 3. **Quantum State Discrimination Techniques**

def plot_quantum_fisher_information():
    theta_values = np.linspace(0, 2 * np.pi, 100)
    psi = np.sin(theta_values)

    d_psi_d_theta = np.cos(theta_values)

    F_Q = 4 * (np.abs(d_psi_d_theta)**2 - np.abs(psi * d_psi_d_theta)**2)

    plt.figure(figsize=(10, 6))
    plt.plot(theta_values, F_Q, label='Quantum Fisher Information $F_Q$', color='purple')
    plt.xlabel('Parameter $theta$', fontsize=14)
    plt.ylabel('$F_Q$', fontsize=14)
    plt.title('Quantum Fisher Information', fontsize=16)
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend()
    plt.show()

# 4. **Sensitivity Measurement**

def plot_sensitivity():
    theta_values = np.linspace(0, 2 * np.pi, 100)
    psi = np.sin(theta_values)
    d_psi_d_theta = np.cos(theta_values)
    F_Q = 4 * (np.abs(d_psi_d_theta)**2 - np.abs(psi * d_psi_d_theta)**2)

    h_min = 1e-22  # minimum strain
    S = h_min / np.sqrt(F_Q)  # Sensitivity calculation

    plt.figure(figsize=(10, 6))
    plt.plot(theta_values, S, label='Sensitivity $S$', color='orange')
    plt.xlabel('Parameter $theta$', fontsize=14)
    plt.ylabel('$S$', fontsize=14)
    plt.title('Sensitivity Measurement $S$', fontsize=16)
    plt.ylim(0, np.max(S) * 1.1)  # Keep some margin
    plt.grid()
    plt.legend()
    plt.show()

# Execute the plotting functions
plot_polarization_squeezing()
plot_fibered_interferometry()
plot_quantum_fisher_information()
plot_sensitivity()
