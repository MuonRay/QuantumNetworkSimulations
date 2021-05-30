
import matplotlib 
matplotlib.use('TkAgg') 
from pylab import * 
import networkx as nx
import random as rd

def initialize(): 
    global g
    g = nx.karate_club_graph()
    for i in g.nodes():
        g.node[i]['state'] = 1 if random() < .5 else 0


def observe(): 
    global g
    cla()
    nx.draw(g, cmap = cm.hsv, vmin = -1, vmax = 1, 
            node_color = [g.node[i]['state'] for i in g.nodes()], 
            pos = nx.spring_layout(g) )


def update(): 
    global g
    listener = rd.choice(g.nodes())
    speaker = rd.choice(g.neighbors(listener))
    g.node[listener]['state'] = g.node[speaker]['state']
    g.add_edge(0,1)
    g[0]['visited'] = True
    g.neighbors(0)
    ['visited', 1]


import pycxsimulator 

pycxsimulator.GUI().start(func=[initialize, observe, update]) 

