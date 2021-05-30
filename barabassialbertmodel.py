# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:01:57 2021

@author: cosmi
"""

import matplotlib 
matplotlib.use('TkAgg') 
from pylab import * 
import networkx as nx 

m0 = 5 # number of nodes in initial condition 
m = 2 # number of edges per new node 

def initialize(): 
    global g, nextg, counter
    g = nx.complete_graph(m0) 
    g.pos = nx.spring_layout(g) 
    nextg = g.copy() 

    
xdata = []    
ydata = []

def observe(): 
    global g, nextg, counter 
    subplot(1,2,1)
    cla() 
    nx.draw(g) 
    
    subplot(1,2,2)
    cla()
    plot(xdata, ydata,'o',alpha = 0.05)
    axis('image')
    # for percolation search

    
def pref_select(nds): 
    global g 
    r = uniform(0, sum(g.degree(i) for i in nds)) 
    x = 0 
    for i in nds: 
        x += g.degree(i) 
        if r <= x: 
            return i 


def update(): 
    global g, nextg, counter
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

#g.pos = nx.spring_layout(pos = g.pos, iterations = 5) 

import pycxsimulator 

pycxsimulator.GUI().start(func=[initialize, observe, update]) 


