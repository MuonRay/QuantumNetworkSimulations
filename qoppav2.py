# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:51:21 2024

@author: ektop
"""

 
import matplotlib 
matplotlib.use('TkAgg') 
from pylab import * 
import networkx as nx
from math import pi
import numpy as np


    
import scipy
import numpy as np
from scipy import misc
import numpy as np
import scipy.linalg as la

from matplotlib import pyplot as plt  # For image viewing

from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm



from random import random as rand
from random import uniform

alpha = 1  # coupling strength
Dt = 0.01  # Delta t

# Define constants for the mass (m) of the oscillators if relevant
m = 1  # Assume mass as 1 for simplicity

# Initialize the network and the next network state

def initialize(): 
    global g, nextg, hilbert_size
    
    

    
    n = 3
    g = nx.grid_graph(dim=[n,n])



#    g = nx.karate_club_graph()
    for i in list(g.nodes()):
        g.node[i]['theta'] = 2 * pi * np.random.random()
        g.node[i]['omega'] = 1. + uniform(-0.05, 0.05)
    nextg = g.copy()

# Visualize the network
def observe(): 
    global g, nextg, grid2d
    subplot(1,2,1)
    cla()
    nx.draw(g, cmap = cm.hsv, vmin = -1, vmax = 1, 
            node_color = [np.sin(g.node[i]['theta']) for i in list(g.nodes())], 
            pos = nx.spring_layout(g) )
    axis('image')
    
    subplot(1,2,2)
    cla()
    
    plot([np.cos(g.node[i]['theta']) for i in list(g.nodes())],
        [np.sin(g.node[i]['theta']) for i in list(g.nodes())], '.')
    axis('image')
    
    axis([-1.1,1.1,-1.1,1.1])
# Define the Gauss/Mouse map transformation
def gauss_mouse_map(phase):
    return np.sin(phase)



# Update the network state
def update():
    global g, nextg, chaotic_numbers_data, timestamps, frequency_shifts, koppa_values, action_derivative_values
    chaotic_numbers = []
    
    num_nodes = len(g.nodes())
    angular_accelerations = np.zeros(num_nodes)
    action_derivative = 0  # Initialize action derivative for this timestep
    
    # Store previous angular velocities
    previous_angular_velocities = np.array([g.nodes[i]['omega'] for i in g.nodes()])
    
    # Create a node to index mapping
    node_to_index = {node: idx for idx, node in enumerate(g.nodes())}
    
    for i in g.nodes():
        idx = node_to_index[i]  # Get the index for the current node
        theta_i = g.nodes[i]['theta']
        omega_i = g.nodes[i]['omega']
        
        # Calculate next angular momentum using Euler's method
        nextg.nodes[i]['theta'] = theta_i + omega_i * Dt + (alpha * (
                np.sum(np.sin(g.nodes[j]['theta'] - theta_i) for j in g.neighbors(i))
                / g.degree(i))) * Dt
        
        # Update angular acceleration
        angular_accelerations[idx] = (nextg.nodes[i]['theta'] - theta_i) / Dt  # This is a simple approximation
        
        chaotic_number = gauss_mouse_map(g.nodes[i]['theta'])
        chaotic_numbers.append(chaotic_number)

    # Now compute the derivative of action
    for i in range(num_nodes):
        action_derivative += 0.5 * m * previous_angular_velocities[i] * angular_accelerations[i]

    action_derivative_values.append(action_derivative)  # Store action derivative over time

    # Calculate frequency shifts
    if len(chaotic_numbers_data) > 0:
        previous_chaotic_numbers = chaotic_numbers_data[-1]
        frequency_shift = [chaotic_numbers[j] - previous_chaotic_numbers[j] for j in range(len(chaotic_numbers))]
        frequency_shifts.append(np.mean(frequency_shift))  # Store the average frequency shift over time
    else:
        frequency_shifts.append(0)  # No shift initially
    
    # Calculate algebraic connectivity (koppa)
    laplacian = nx.laplacian_matrix(g).toarray()
    eigenvalues = np.linalg.eigvals(laplacian)
    koppa = np.sort(eigenvalues)[1]  # Second smallest eigenvalue
    koppa_values.append(koppa)

    g, nextg = nextg, g
    chaotic_numbers_data.append(chaotic_numbers)
    timestamps.append(len(chaotic_numbers_data))

# Initialize and update the network state
def initialize_and_update():
    initialize()
    update()

import pycxsimulator

# Run the simulation
chaotic_numbers_data = []
frequency_shifts = []
timestamps = []
koppa_values = []
action_derivative_values = []  # List to store action derivatives over time
pycxsimulator.GUI().start(func=[initialize, observe, update])

# Create scatter plot of chaotic number values vs timestamps
plt.figure(figsize=(12, 5))
for i, chaotic_numbers in enumerate(chaotic_numbers_data):
    colors = ['r' if num >= 0 else 'b' for num in chaotic_numbers]
    plt.scatter([timestamps[i]] * len(chaotic_numbers), chaotic_numbers, color=colors, alpha=0.5)

plt.xlabel('Timestamp')
plt.ylabel('Chaotic Number Value')
plt.title('Scatter Plot of Chaotic Number Values vs Timestamp')
plt.show()

# Create scatter plot of frequency shifts
plt.figure(figsize=(12, 5))
for i, shift in enumerate(frequency_shifts):
    plt.scatter(timestamps[i], shift, color='g', alpha=0.5)  # Use just `timestamps[i]` for y values

plt.xlabel('Timestamp')
plt.ylabel('Average Frequency Shift')
plt.title('Scatter Plot of Frequency Shifts vs Timestamp')
plt.show()

# Plot the trend of wavelength shifts vs koppa
plt.figure(figsize=(12, 5))
plt.plot(timestamps, koppa_values, marker='o', linestyle='-')
plt.title('Algebraic Connectivity Koppa over Time')
plt.xlabel('Timestamp')
plt.ylabel('Algebraic Connectivity (Koppa)')
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.scatter(koppa_values, frequency_shifts, color='purple', alpha=0.5)
plt.title('Wavelength Shifts vs Algebraic Connectivity Koppa')
plt.xlabel('Algebraic Connectivity (Koppa)')
plt.ylabel('Average Frequency Shift')
plt.grid()
plt.show()

# New: Plot action derivatives over time
plt.figure(figsize=(12, 5))
plt.plot(timestamps, action_derivative_values, marker='o', linestyle='-')
plt.title('Action Derivative over Time')
plt.xlabel('Timestamp')
plt.ylabel('Action Derivative')
plt.grid()
plt.show()

# New: Plot frequency shifts vs action derivatives
plt.figure(figsize=(12, 5))
plt.scatter(action_derivative_values, frequency_shifts, color='orange', alpha=0.5)
plt.title('Frequency Shifts vs Action Derivative')
plt.xlabel('Action Derivative')
plt.ylabel('Average Frequency Shift')
plt.grid()
plt.show()