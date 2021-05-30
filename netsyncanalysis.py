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


    
import scipy
import numpy as np
from scipy import misc

from matplotlib import pyplot as plt  # For image viewing

from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap





from random import random as rand
from random import uniform



#def gridsize(val):
#    '''
#    Number Of Particles in a Grid Shall Be Entered such that gridsize 4 = 4 x 4 i.e. 16
#    Particles in Total. Note: this can only be changed at the start of a new 
#    Simulation Run - In This Version Do Note Change While Running the Simulation!
#    '''
#    global n

#    n = int(val)
#    return val
    
    

def initialize(): 
    global g, nextg
    
    n = 5
    g = nx.grid_graph(dim=[n,n])

    #g = nx.karate_club_graph()
    
    for i in list(g.nodes()):
        g.node[i]['theta'] = 2 * pi * random()
        #rows, cols = (-0.05, 0.05)
        #arr = [[rand.randrange(10) for i in range(int(cols))] for j in range(int(rows))]
        #a = numpy.asarray(arr)
        #g.node[i]['omega'] = 1. + rand.uniform(-0.05, 0.05) 
        g.node[i]['omega'] = 1. + uniform(-0.05, 0.05)        
    nextg = g.copy()     
    
        
    for i in list(g.nodes()):
        g.node[i]['theta'] = random()    
    nextg = g.copy() 
   
        
#for space vs time graph
    
xdata = []    
ydata = []


def observe(): 
    global g, nextg
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
   
    
    #subplot(1,2,2)
    #cla()
    #plot(xdata, ydata,'o',alpha = 0.05)
    #axis('image')
    # for space vs time plotting (chimera search)

    
alpha = 2 # coupling strength 
beta = 1 # acceleration rate
Dt = 0.01 # Delta t 

#def update(): 
#    global g, nextg 
#    for i in list(g.nodes()): 
#        theta_i = g.node[i]['theta'] 
#        nextg.node[i]['theta'] = theta_i + (beta * theta_i + alpha * (np.sum(sin(g.node[j]['theta'] - theta_i) for j in g.neighbors(i))) * Dt) 
#    g, nextg = nextg, g 

def update(): 
    global g, nextg 
    for i in list(g.nodes()): 
        theta_i = g.node[i]['theta'] 
        nextg.node[i]['theta'] = theta_i + (g.node[i]['omega'] + alpha * ( \
           sum(np.sin(g.node[j]['theta'] - theta_i) for j in g.neighbors(i)) \
           / g.degree(i))) * Dt 
    g, nextg = nextg, g 
    
    
    #for i, j in list(g.nodes()):
        #xdata.append(g.degree(i))
        #ccs = nx.connected_components(g)
        #ydata.append(max(len(cc) for cc in ccs))
        #xdata.append(g.degree(i)); ydata.append(g.degree(j))
        #xdata.append(g.degree(j)); ydata.append(g.degree(i))



import pycxsimulator 

pycxsimulator.GUI().start(func=[initialize, observe, update]) 



#plt.figure(1)
    #compare red and blue pixel data
#nbins = 20
#plt.hexbin(x=plot_time_stamp, y=plot_agent, gridsize=nbins, cmap=plt.cm.jet)
#plt.xlabel('Blue Reflectance')
#plt.ylabel('NIR Reflectance')
    # Add a title
#plt.title('NIR vs Blue Spectral Data')
#plt.show()
