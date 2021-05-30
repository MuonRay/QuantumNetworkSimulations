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

import time # for steptime

# for space vs time plotting (chimera search)
    
import scipy
import numpy as np
from scipy import misc

from matplotlib import pyplot as plt  # For image viewing

from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap





from random import random as rand
from random import uniform

def initialize(): 
    global g, nextg, counter 
    s = 5
    g = nx.grid_graph(dim=[s,s])
    #nodes = list(G.nodes)
    #edges = list(G.edges)

    #g = nx.karate_club_graph()
    counter = 0
    for i in list(g.nodes()):
        g.node[i]['theta'] = 2 * pi * random()
        #rows, cols = (-0.05, 0.05)
        #arr = [[rand.randrange(10) for i in range(int(cols))] for j in range(int(rows))]
        #a = numpy.asarray(arr)
        #g.node[i]['omega'] = 1. + rand.uniform(-0.05, 0.05) 
        g.node[i]['omega'] = 1. + uniform(-0.05, 0.05)        
    nextg = g.copy() 
    counter = +1

def observe(): 
    global g, nextg
    cla()
    nx.draw(g, cmap = cm.hsv, vmin = -1, vmax = 1, 
            node_color = [np.sin(g.node[i]['theta']) for i in list(g.nodes())], 
            pos = nx.spring_layout(g) )
    
   
alpha = 1 # coupling strength 
Dt = 0.01 # Delta t 

def update(): 
    global g, nextg 
    for i in list(g.nodes()): 
        theta_i = g.node[i]['theta'] 
        nextg.node[i]['theta'] = theta_i + (g.node[i]['omega'] + alpha * ( \
           sum(np.sin(g.node[j]['theta'] - theta_i) for j in g.neighbors(i)) \
           / g.degree(i))) * Dt 
    g, nextg = nextg, g 
    
    agents = theta_i
    """
environment
"""

    # empty numpy array for environmental state
    plot_time_stamp = []
    plot_agent = []
    
        # save for figure
    plot_time_stamp.append(counter)
    plot_agent.append(agents)
        





import pycxsimulator 

pycxsimulator.GUI().start(func=[initialize, observe, update]) 


plt.figure(1)
    #compare red and blue pixel data
nbins = 20
plt.hexbin(x=plot_time_stamp, y=plot_agent, gridsize=nbins, cmap=plt.cm.jet)
plt.xlabel('Blue Reflectance')
plt.ylabel('NIR Reflectance')
    # Add a title
plt.title('NIR vs Blue Spectral Data')
plt.show()
