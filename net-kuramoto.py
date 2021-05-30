# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 13:30:10 2021

@author: cosmi
"""


import matplotlib 
matplotlib.use('TkAgg') 
from pylab import * 
import networkx as nx
from math import sin, pi
import numpy

from random import random as rand


def initialize(): 
    global g, nextg 
    g = nx.karate_club_graph()
    g.pos = nx.spring_layout(g) 
    
    
    for i in list(g.nodes()):

    #for i in g.nodes_iter(): 
        g.node[i]['theta'] = 2*pi*rand() 
        
        rows, cols = (-0.05, 0.05)
        arr = [[rand.randrange(10) for i in range(int(cols))] for j in range(int(rows))]
        a = numpy.asarray(arr)
        #g.node[i]['omega'] = 1. + rand.uniform(-0.05, 0.05) 
        g.node[i]['omega'] = 1. + a
        
        nextg = g.copy() 

def observe(): 
    global g, nextg
    nx.draw(g, cmap = cm.hsv, vmin = -1, vmax = 1, 
            node_color = [sin(g.node[i]['theta']) for i in list(g.nodes())], 
            pos = g.pos) 

alpha = 1 # coupling strength 
Dt = 0.01 # Delta t 

def update(): 
    global g, nextg 
    for i in list(g.nodes()): 
        theta_i = g.node[i]['theta'] 
        nextg.node[i]['theta'] = theta_i + (g.node[i]['omega'] + alpha * ( \
        sum(sin(g.node[j]['theta'] - theta_i) for j in g.neighbors(i))  
         /g.degree(i))) * Dt 
    g, nextg = nextg, g 


import pycxsimulator 

pycxsimulator.GUI().start(func=[initialize, observe, update]) 
