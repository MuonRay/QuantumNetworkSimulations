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
import numpy as np
import scipy.linalg as la

from matplotlib import pyplot as plt  # For image viewing

from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm



from random import random as rand
from random import uniform

from qutip.visualization import plot_wigner, hinton

from pygsp import graphs

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
    
    n = 3
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
    
    
    
    
    

    
       
    grid2d = graphs.Graph.from_networkx(nextg)
    
    print(grid2d.W.toarray())
    print(grid2d.signals)
    print(grid2d)
    
    grid2d.compute_fourier_basis()
    
    grid2d.set_coordinates()
    




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

        
#for space vs time graph
    
xdata = []    
ydata = []


def observe(): 
    global g, nextg, grid2d
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
    global g, nextg, eig_values, eig_vectors, rho, grid2d
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


    A = nx.adjacency_matrix(nextg)
    print(A)
    n, m = A.shape
    diags = A.sum(axis=0)  # 1 = outdegree, 0 = indegree
    D = scipy.sparse.spdiags(diags.flatten(), [0], m, n, format="csr")
    L = (A-D)
    Lap = L.todense()
    print(Lap)
    
    eig_values, eig_vectors = la.eig(Lap)
    fiedler_pos = np.where(eig_values.real == np.sort(eig_values.real)[1])[0][0]
    fiedler_vector = np.transpose(eig_vectors)[fiedler_pos]
    
    print("Fiedler value: " + str(fiedler_pos.real))

    print("Fiedler vector: " + str(fiedler_vector.real))
    #nx.laplacian_matrix(nextg).toarray()
    
    
    # applying matrix.trace() method
    LTrace = np.matrix.trace(Lap)
    print(LTrace)
    
    #print density matrix 
    rho = np.divide(Lap,LTrace)
    print(rho)
    
    
    
    
    
 

    

    #note you can calculate the trace faster using the hadamard product (element-wise multiplication)
    # using the fiedler vector as the basis for the emergent density matrix 
    
    


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
