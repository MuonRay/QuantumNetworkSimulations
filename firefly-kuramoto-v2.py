# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 13:30:10 2021

@author: cosmi
"""


import matplotlib 
matplotlib.use('TkAgg') 
from pylab import * 
import networkx as nx
from math import pi
import numpy as np

from random import random as rand
from random import uniform

def initialize(): 
    global g, nextg 
    g = nx.karate_club_graph()
    for i in list(g.nodes()):
        g.node[i]['theta'] = 2 * pi * random()
        #rows, cols = (-0.05, 0.05)
        #arr = [[rand.randrange(10) for i in range(int(cols))] for j in range(int(rows))]
        #a = numpy.asarray(arr)
        #g.node[i]['omega'] = 1. + rand.uniform(-0.05, 0.05) 
        g.node[i]['omega'] = 1. + uniform(-0.05, 0.05)        
    nextg = g.copy() 

def observe(): 
    global g, nextg
    cla()
    nx.draw(g, cmap = cm.hsv, vmin = -1, vmax = 1, 
            node_color = [np.sin(g.node[i]['theta']) for i in list(g.nodes())], 
            pos = nx.spring_layout(g) )
    
    fig = go.Figure(data=go.Scatter(x=plot_time_stamp, y=plot_agent, mode='markers',
                                    marker=dict(size=4.5, color="Blue", opacity=0.6)))
    fig.show()

alpha = 1 # coupling strength 
Dt = 0.01 # Delta t 

def update(): 
    global g, nextg 
    for i in list(g.nodes()): 
        theta_i = g.node[i]['theta'] 
        nextg.node[i]['theta'] = theta_i + (g.node[i]['omega'] + alpha * ( \
           np.sum(np.sin(g.node[j]['theta'] - theta_i) for j in g.neighbors(i)) \
           / g.degree(i))) * Dt 
    g, nextg = nextg, g 


import pycxsimulator 

pycxsimulator.GUI().start(func=[initialize, observe, update]) 



