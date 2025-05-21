# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 19:59:18 2025

@author: ektop
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:26:44 2024

@author: ektop
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from itertools import combinations

# Define the number of nodes in each component
NUM_NODES_A = 10  # Number of nodes in group A
NUM_NODES_B = 10  # Number of nodes in group B

# Create a structured directed acyclic graph (DAG)
dag = nx.DiGraph()

def create_random_bipartite_structure(num_nodes_a, num_nodes_b):
    """Create a random bipartite-like structure."""
    edges = [(i, num_nodes_a + j) for i in range(num_nodes_a) for j in range(num_nodes_b)]
    random.shuffle(edges)
    selected_edges = random.sample(edges, k=random.randint(1, len(edges)))
    return selected_edges

# Add random edges to the graph
dag.add_edges_from(create_random_bipartite_structure(NUM_NODES_A, NUM_NODES_B))
dag.add_edges_from(create_random_bipartite_structure(NUM_NODES_A, NUM_NODES_B))

# Generate connections between the nodes
component_connections = []
connections_from_first_to_second = random.sample(
    [(i, NUM_NODES_A + j) for i in range(NUM_NODES_A) for j in range(NUM_NODES_B)], 
    k=random.randint(1, 2))
component_connections.extend(connections_from_first_to_second)

# Create additional connections for the second section of nodes
second_section_nodes = list(range(NUM_NODES_A, NUM_NODES_A + NUM_NODES_B))
new_node = NUM_NODES_A + NUM_NODES_B
for node in second_section_nodes:
    additional_connections = random.sample([new_node] + second_section_nodes, 
                                           k=random.randint(1, len(second_section_nodes))) 
    component_connections += [(node, conn) for conn in additional_connections if conn != node]

dag.add_edges_from(component_connections)

def compute_cheeger_constant(G):
    """Calculate the Cheeger constant of the graph."""
    n = len(G.nodes)
    cuts = []
    for cut_nodes in combinations(range(n), n // 2):
        cut_size = len([edge for edge in G.edges if (edge[0] in cut_nodes) ^ (edge[1] in cut_nodes)])
        cuts.append(cut_size)
    return min(cuts) / min(sum(1 for node in cut_nodes), sum(1 for node in G.nodes if node not in cut_nodes))

# Calculate the initial Cheeger constant
initial_cheeger_constant = compute_cheeger_constant(dag)
print(f"Initial Cheeger Constant: {initial_cheeger_constant}")

def laplacian_matrix(A):
    """Compute the normalized Laplacian matrix."""
    D = np.diag(np.sum(A, axis=1))
    return D - A

def effective_resistance(u, v, eigenvalues, fiedler_vector):
    """Calculate the effective resistance between two nodes."""
    if len(eigenvalues) == 0 or len(fiedler_vector) == 0:
        return np.inf 
    sum_term = 0
    for i in range(len(eigenvalues)):
        if eigenvalues[i] > 0: 
            sum_term += (fiedler_vector[u] * fiedler_vector[v]) / eigenvalues[i]
    return sum_term

def optimize_graph(g, iterations=100):
    """Optimize the graph to minimize effective resistance and maximize Cheeger constant."""
    previous_cheeger_constant = compute_cheeger_constant(g)
    
    for _ in range(iterations):
        # Step 1: Remove a random edge
        if len(g.edges) > 0:
            edge = random.choice(list(g.edges))
            g.remove_edge(*edge)
        
            # Step 2: Add a new random edge
            possible_edges = [(i, j) for i in range(len(g.nodes)) for j in range(len(g.nodes)) 
                              if i != j and not g.has_edge(i, j)]
            if possible_edges:
                new_edge = random.choice(possible_edges)
                g.add_edge(*new_edge)
        
        # Step 3: Calculate the new Cheeger constant
        new_cheeger_constant = compute_cheeger_constant(g)
        print(f"Current Cheeger Constant: {new_cheeger_constant}")

        # Check for convergence
        if new_cheeger_constant == previous_cheeger_constant:
            break
        
        previous_cheeger_constant = new_cheeger_constant

# Run the optimization process
optimize_graph(dag)

# Final graph attributes and effective resistance calculation
num_nodes = dag.number_of_nodes()
adjacency = nx.to_numpy_array(dag)
normalized_laplacian = laplacian_matrix(adjacency)

# Eigenvalue decomposition for effective resistance calculation
eigenvalues, eigenvectors = np.linalg.eig(normalized_laplacian)
order = np.argsort(eigenvalues)
fiedler_vector = np.real(eigenvectors[:, order[1]])

# Compute effective resistances for all node pairs
effective_resistances = np.zeros((num_nodes, num_nodes))
for u, v in combinations(range(num_nodes), 2):
    effective_resistances[u, v] = effective_resistance(u, v, eigenvalues[order[1:]], fiedler_vector)
    effective_resistances[v, u] = effective_resistances[u, v]

# Visualization of effective resistances
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(dag)  # Positioning for the nodes
node_color = [np.mean(effective_resistances[node]) for node in range(num_nodes)]

# Draw the graph with effective resistances represented as colors
nodes = nx.draw(dag, pos, node_color=node_color, with_labels=True, cmap=plt.cm.viridis, node_size=500)

# Create the ScalarMappable for color representation
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
sm.set_array(node_color)

# Add colorbar to the plot
plt.colorbar(sm, label='Effective Resistance Level')
plt.title('Graph Visualization of Effective Resistances')
plt.show()
