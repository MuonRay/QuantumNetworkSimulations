# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:15:19 2024

@author: ektop
"""

import networkx as nx
import numpy as np
from random import uniform
from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

alpha = 1  # coupling strength
Dt = 0.01  # Delta t
m = 1  # Assume mass as 1 for simplicity

def initialize():
    global g, nextg
    g = nx.karate_club_graph()
    for i in list(g.nodes()):
        g.node[i]['theta'] = 2 * pi * np.random.random()
        g.node[i]['omega'] = 1. + uniform(-0.05, 0.05)
    nextg = g.copy()

def observe():
    global g
    plt.clf()
    nx.draw(g, cmap=plt.cm.hsv, vmin=-1, vmax=1,
            node_color=[np.sin(g.node[i]['theta']) for i in list(g.nodes())],
            pos=nx.spring_layout(g))
    plt.title('Network Visualization')
    plt.show()

def gauss_mouse_map(phase):
    return np.sin(phase)

def update():
    global g, nextg, chaotic_numbers_data, timestamps, frequency_shifts, action_derivative_values
    chaotic_numbers = []
    angular_accelerations = np.zeros(len(g.nodes()))
    action_derivative = 0  # Initialize action derivative for this timestep
    previous_angular_velocities = np.array([g.node[i]['omega'] for i in g.nodes()])

    for i in list(g.nodes()):
        theta_i = g.node[i]['theta']
        omega_i = g.node[i]['omega']
        
        # Update angular position using Euler's method
        nextg.node[i]['theta'] = theta_i + omega_i * Dt + (alpha * (
                np.sum(np.sin(g.node[j]['theta'] - theta_i) for j in g.neighbors(i))
                / g.degree(i))) * Dt
        
        angular_accelerations[i] = (nextg.node[i]['theta'] - theta_i) / Dt
        
        chaotic_number = gauss_mouse_map(g.node[i]['theta'])
        chaotic_numbers.append(chaotic_number)

    # Compute derivative of action
    for i in range(len(g.nodes())):
        action_derivative += 0.5 * m * previous_angular_velocities[i] * angular_accelerations[i]

    action_derivative_values.append(action_derivative)  # Store action derivative over time
        # Calculate frequency shifts
    if len(chaotic_numbers_data) > 0:
        previous_chaotic_numbers = chaotic_numbers_data[-1]
        frequency_shift = [chaotic_numbers[j] - previous_chaotic_numbers[j] for j in range(len(chaotic_numbers))]
        frequency_shifts.append(np.mean(frequency_shift))  # Store the average frequency shift over time
    else:
        frequency_shifts.append(0)  # No shift initially
    

    # Update the states
    g, nextg = nextg, g
    chaotic_numbers_data.append(chaotic_numbers)
    timestamps.append(len(chaotic_numbers_data))

def initialize_and_update():
    initialize()
    update()

import pycxsimulator

# Initialize lists to store data
chaotic_numbers_data = []
frequency_shifts = []
action_derivative_values = []  # List to store action derivatives over time
timestamps = []  # Initialize timestamps

# Run the simulation
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
    plt.scatter(timestamps[i], shift, color='g', alpha=0.5)

plt.xlabel('Timestamp')
plt.ylabel('Average Frequency Shift')
plt.title('Scatter Plot of Frequency Shifts vs Timestamp')
plt.show()


# New: Plot action derivatives over time
plt.figure(figsize=(12, 5))
plt.plot(timestamps, action_derivative_values, marker='o', linestyle='-')
plt.title('Action Derivative over Time')
plt.xlabel('Timestamp')
plt.ylabel('Action Derivative')
plt.grid()
plt.show()

# New: Create scatter plot of action derivative vs chaotic numbers
plt.figure(figsize=(12, 5))
for i, chaotic_numbers in enumerate(chaotic_numbers_data):
    for j in range(len(chaotic_numbers)):
        plt.scatter(action_derivative_values[i], chaotic_numbers[j], color='red', alpha=0.5)

plt.title('Chaotic Numbers vs Action Derivative')
plt.xlabel('Action Derivative')
plt.ylabel('Chaotic Number Value')
plt.grid()
plt.show()


# New: Plot frequency shifts vs actions
plt.figure(figsize=(12, 5))
plt.scatter(action_derivative_values, frequency_shifts, color='orange', alpha=0.5)
plt.title('Frequency Shifts vs Action Derivative')
plt.xlabel('Action Derivative')
plt.ylabel('Average Frequency Shift')
plt.grid()
plt.show()





# Assuming you have the following variables defined
# timestamps, action_derivative_values, frequency_shifts

# Create a figure
fig = plt.figure(figsize=(12, 8))

# Create a 3D scatter plot
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot, using timestamps, action derivatives, and chaotic numbers
scatter = ax.scatter(timestamps, action_derivative_values, frequency_shifts, 
                     c=action_derivative_values, cmap='viridis', alpha=0.5)

# Add titles and labels
ax.set_title('3D Visualization of Action Derivative, Frequency Shift, va Timestamps')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Action Derivative')
ax.set_zlabel('Frequency Shift Value')

# Show color bar for reference
plt.colorbar(scatter, label='Action Derivative')

# Show the plot
plt.show()




