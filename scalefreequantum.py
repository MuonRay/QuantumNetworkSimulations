# -*- coding: utf-8 -*-
"""
Created on Sun May 28 01:18:22 2023

@author: ektop
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:01:57 2021

@author: cosmi
"""

import matplotlib 
matplotlib.use('TkAgg') 
from pylab import * 
import networkx as nx 

import numpy as np

from pygsp import graphs
import matplotlib.pyplot as plt

m0 = 4 # number of nodes in initial condition 
m = 2 # number of edges per new node 

global grid2d

counter = 0

def initialize(): 
    global g, nextg, counter, grid2d
    g = nx.complete_graph(m0) 
    g.pos = nx.spring_layout(g) 
    nextg = g.copy() 

    
xdata = []    
ydata = []

grid2d = []

def observe1(): 
    global g, nextg, counter, grid2d
    

    
    subplot(1,2,1)
    cla() 
    nx.draw(g) 
    
    #subplot(1,2,2)
    #cla()
    #plot(xdata, ydata,'o',alpha = 0.05)
    #axis('image')
    

def observe2(): 
    global g, nextg, counter, grid2d
    
    subplot(1,2,2)
    grid2d = graphs.Graph.from_networkx(g)

    plt.imshow(grid2d.A.todense())
    axis('image')



    
def pref_select(nds): 
    global g 
    r = uniform(0, sum(g.degree(i) for i in nds)) 
    x = 0 
    for i in nds: 
        x += g.degree(i) 
        if r <= x: 
            return i 


def update(): 
    global g, nextg, counter, grid2d
    counter += 1
    if counter % 20 == 0:
        nds = g.nodes()
        newcomer = max(nds) + 1 
        
        for i in range(m): 
            j = pref_select(nds) 
            g.add_edge(newcomer, j) 
            unsaturated_b = g.nodes()
            list(unsaturated_b).remove(j)
            
            xdata.append(g.degree(i))
            ccs = nx.connected_components(g)
            ydata.append(max(len(cc) for cc in ccs))
        #xdata.append(g.degree(i)); ydata.append(g.degree(j))
        #xdata.append(g.degree(j)); ydata.append(g.degree(i))
        #g.pos[newcomer] = (0, 0) # simulation of node movement 
        g, nextg = nextg, g
        
        grid2d = graphs.Graph.from_networkx(g)
        


#g.pos = nx.spring_layout(pos = g.pos, iterations = 5) 

import pycxsimulator2plots 

pycxsimulator2plots.GUI().start(func=[initialize, observe1, observe2, update]) 






# for percolation search at end of run
pycxsimulator2plots.GUI().quitGUI


print(grid2d.W.toarray())
print(grid2d.signals)

print(grid2d)



grid2d.compute_fourier_basis()

grid2d.set_coordinates()
grid2d.plot()

plt.imshow(grid2d.A.todense())



# plot spectrum
fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.plot(grid2d.e)
ax.set_xlabel('eigenvalue index (i)')
ax.set_ylabel('eigenvalue ($\lambda_{i}$)')
ax.set_title('2D-grid spectrum');


#fiedler vector highlighted graph
grid2d.plot_signal(grid2d.U[:,1])


#plot all eigenvectors as network graph frames

fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
count = 0
for j in range(2):
    for i in range(3):
        grid2d.plot_signal(grid2d.U[:, count*1], ax=axes[j,i],colorbar=False)
        axes[j,i].set_xticks([])
        axes[j,i].set_yticks([])
        axes[j,i].set_title(f'Eigvec {count*1+1}')
        count+=1
fig.tight_layout()


    
    
    
    