# Epidemic model

import matplotlib 
matplotlib.use('TkAgg') 
from pylab import * 
import networkx as nx
import random as rd

def initialize(): 
    global g, nextg
    g = nx.karate_club_graph()
    for i in g.nodes():
        g.node[i]['state'] = 1 if random() < .5 else 0
    nextg = g.copy() 



def observe(): 
    global g, nextg
    cla()
    nx.draw(g, cmap = cm.hsv, vmin = -1, vmax = 1, 
            node_color = [g.node[i]['state'] for i in g.nodes()], 
            pos = nx.spring_layout(g) )

p_i = 0.5 # infection probability
p_r = 0.5 # recovery probability

def update(): 
    global g, nextg
    a = rd.choice(g.nodes())
    if g.node[a]['state'] == 0: # if susceptable to infection
        b = rd.choice(g.neighbors(a))
        if g.node[b]['state'] == 1: # if neighbor b is infected
            g.node[a]['state'] = 1 if random() < p_i else 0
            
        else: # if infected
            g.node[a]['state'] = 1 if random() < p_r else 1
            
            


    
    #g.add_edge(0,1)
    #g[0]['visited'] = True
    #g.neighbors(0)
    #['visited', 1]


import pycxsimulator 

pycxsimulator.GUI().start(func=[initialize, observe, update]) 

