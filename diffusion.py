
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

alpha = 1 # coupling strength 
Dt = 0.01 # Delta t 

def update(): 
    global g, nextg 
    for i in list(g.nodes()): 
        ci = g.node[i]['state'] 
        nextg.node[i]['state'] = ci + alpha * ( \
           np.sum(g.node[j]['state'] for j in g.neighbors(i)) \
           -ci * g.degree(i)) * Dt 
    g, nextg = nextg, g 


    
    g.add_edge(0,1)
    g[0]['visited'] = True
    g.neighbors(0)
    ['visited', 1]


import pycxsimulator 

pycxsimulator.GUI().start(func=[initialize, observe, update]) 

