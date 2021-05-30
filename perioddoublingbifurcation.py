# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:28:58 2021

@author: cosmi
"""

from pylab import * 
def initialize(): 
    global x, result 
    x = 0.1 
    result = [x] 


def observe(): 
    global x, result
    result.append(x)
    
def update():
    global x, result
    x = x + r - x**2

#def plot_asymptotic_states():
def plot_phase_space():
    initialize()
    for t in range(30): 
        update() 
        observe() 
    plot(result) 
    ylim(0, 2) 
    title('r = ' + str(r)) 
rs = [0.1, 0.5, 1.0, 1.1, 1.5, 1.6] 

for i in range(len(rs)): 
    subplot(2, 3, i + 1)
    r = rs[i]
    plot_phase_space() 
show() 
