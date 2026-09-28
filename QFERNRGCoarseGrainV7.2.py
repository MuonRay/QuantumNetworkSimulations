# -*- coding: utf-8 -*-
"""
QFERN + Spectral Renormalisation Group
======================================

Generalised multi-cluster implementation.

This version allows the user to specify:

    1. Number of microscopic clusters
    2. Number of nodes in each cluster
    3. Topology of each cluster:
          - strogatz
          - barabasi_albert
          - karate_club
          - star
          - ring
          - line

The resulting network is analysed using:

    QFERN
        |
        v
    Laplacian
        |
        v
    Fiedler / low-frequency spectral structure
        |
        v
    Spectral clustering
        |
        v
    Coarse-graining matrix P
        |
        v
    L^(1) = P.T L^(0) P
        |
        v
    Coarse pseudoinverse
        |
        v
    Effective resistance
        |
        v
    Linearised Kuramoto response
        |
        v
    RG flow

Resistance analysis is performed at both scales:

    Scale 0:
        R^(0) = effective resistance of microscopic graph

    Scale 1:
        R^(1) = effective resistance of coarse graph

A comparison figure is also produced using the cluster-averaged
microscopic resistance:

    Rbar^(0)_ab =
        average resistance between nodes in clusters C_a and C_b

and

    Delta R_ab =
        R^(1)_ab - Rbar^(0)_ab

Plots are automatically saved to a dedicated results directory
using stable filenames suitable for inclusion in an IEEE/Overleaf
paper.

Primary output files:

    01_micro_graph.png
    02_fiedler_vector.png
    03_effective_resistance.png
    03b_renormalised_effective_resistance.png
    03c_effective_resistance_comparison.png
    04_coarse_graph.png
    05_coarse_fiedler_vector.png
    06_rg_flow.png

Additional outputs:

    07_eigenvalue_spectrum.png
    08_qfern_edge_sensitivity.png
    09_kuramoto_response.png

Numerical data:

    qfern_rg_data.npz

Text summary:

    simulation_summary.txt
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# GLOBAL SETTINGS
# ============================================================

DEFAULT_SEED = 42

RESULTS_ROOT = "qfern_rg_results"


# ============================================================
# 1. CREATE CLUSTER TOPOLOGIES
# ============================================================

def create_cluster(
        n,
        topology,
        seed=42,
        cluster_id=0):

    """
    Create a single microscopic cluster.

    Supported topologies:

        strogatz
        barabasi_albert
        karate_club
        star
        ring
        line
    """

    topology = topology.lower().strip()

    # --------------------------------------------------------
    # Star
    # --------------------------------------------------------

    if topology == "star":

        if n < 2:
            raise ValueError(
                "A star requires at least 2 nodes."
            )

        G = nx.star_graph(n - 1)

    # --------------------------------------------------------
    # Ring
    # --------------------------------------------------------

    elif topology == "ring":

        if n < 3:
            raise ValueError(
                "A ring requires at least 3 nodes."
            )

        G = nx.cycle_graph(n)

    # --------------------------------------------------------
    # Line
    # --------------------------------------------------------

    elif topology == "line":

        if n < 2:
            raise ValueError(
                "A line requires at least 2 nodes."
            )

        G = nx.path_graph(n)

    # --------------------------------------------------------
    # Watts-Strogatz
    # --------------------------------------------------------

    elif topology == "strogatz":

        if n < 4:
            raise ValueError(
                "A Watts-Strogatz graph requires "
                "at least 4 nodes."
            )

        k = min(4, n - 1)

        if k % 2 != 0:
            k -= 1

        if k < 2:
            k = 2

        G = nx.watts_strogatz_graph(
            n=n,
            k=k,
            p=0.25,
            seed=seed
        )

    # --------------------------------------------------------
    # Barabasi-Albert
    # --------------------------------------------------------

    elif topology == "barabasi_albert":

        if n < 3:
            raise ValueError(
                "Barabasi-Albert requires at least 3 nodes."
            )

        m = min(2, n - 1)

        G = nx.barabasi_albert_graph(
            n=n,
            m=m,
            seed=seed
        )

    # --------------------------------------------------------
    # Karate Club
    # --------------------------------------------------------

    elif topology == "karate_club":

        base = nx.karate_club_graph()

        if n != base.number_of_nodes():

            raise ValueError(
                "The Karate Club graph contains exactly "
                f"{base.number_of_nodes()} nodes. "
                f"You requested {n}."
            )

        G = base.copy()

    else:

        raise ValueError(
            f"Unknown topology '{topology}'.\n"
            "Choose from: strogatz, barabasi_albert, "
            "karate_club, star, ring, line."
        )

    # Ensure nodes start at zero.
    G = nx.convert_node_labels_to_integers(G)

    # Add edge weights.
    for u, v in G.edges():

        G[u][v]["weight"] = 1.0

    return G


# ============================================================
# 2. CREATE MULTI-CLUSTER NETWORK
# ============================================================

def create_clustered_network(
        cluster_sizes,
        topologies,
        bridge_type="chain",
        seed=42):

    """
    Construct a network containing multiple microscopic
    clusters.
    """

    if len(cluster_sizes) != len(topologies):

        raise ValueError(
            "cluster_sizes and topologies must "
            "have the same length."
        )

    rng = np.random.default_rng(seed)

    G = nx.Graph()

    clusters = []

    node_offset = 0

    # --------------------------------------------------------
    # Construct individual clusters
    # --------------------------------------------------------

    for cluster_id, (size, topology) in enumerate(
            zip(cluster_sizes, topologies)):

        local_seed = int(
            rng.integers(0, 1_000_000)
        )

        local_graph = create_cluster(
            size,
            topology,
            seed=local_seed,
            cluster_id=cluster_id
        )

        mapping = {
            node: node + node_offset
            for node in local_graph.nodes()
        }

        local_graph = nx.relabel_nodes(
            local_graph,
            mapping
        )

        G.add_nodes_from(
            local_graph.nodes()
        )

        G.add_edges_from(
            local_graph.edges(data=True)
        )

        cluster_nodes = list(
            local_graph.nodes()
        )

        clusters.append(
            cluster_nodes
        )

        node_offset += size

    # --------------------------------------------------------
    # Connect clusters
    # --------------------------------------------------------

    if len(clusters) > 1:

        if bridge_type == "chain":

            for i in range(len(clusters) - 1):

                connect_clusters(
                    G,
                    clusters[i],
                    clusters[i + 1]
                )

        elif bridge_type == "ring":

            for i in range(len(clusters)):

                j = (i + 1) % len(clusters)

                connect_clusters(
                    G,
                    clusters[i],
                    clusters[j]
                )

        elif bridge_type == "all":

            for i in range(len(clusters)):

                for j in range(i + 1, len(clusters)):

                    connect_clusters(
                        G,
                        clusters[i],
                        clusters[j]
                    )

        else:

            raise ValueError(
                "bridge_type must be "
                "'chain', 'ring', or 'all'."
            )

    return G, clusters


# ============================================================
# 3. CONNECT TWO CLUSTERS
# ============================================================

def connect_clusters(
        G,
        cluster_a,
        cluster_b):

    """
    Add a single inter-cluster bridge.

    The selected nodes are the final node of cluster A
    and first node of cluster B.
    """

    node_a = cluster_a[-1]
    node_b = cluster_b[0]

    G.add_edge(
        node_a,
        node_b,
        weight=1.0
    )


# ============================================================
# 4. LAPLACIAN
# ============================================================

def compute_laplacian(G):

    L = nx.laplacian_matrix(
        G,
        weight="weight"
    ).toarray().astype(float)

    return L


# ============================================================
# 5. SPECTRAL DECOMPOSITION
# ============================================================

def spectral_decomposition(L):

    if L.ndim != 2:
        raise ValueError(
            "Laplacian must be a matrix."
        )

    if L.shape[0] != L.shape[1]:
        raise ValueError(
            "Laplacian must be square."
        )

    eigvals, eigvecs = np.linalg.eigh(L)

    return eigvals, eigvecs


# ============================================================
# 6. FIEDLER VECTOR
# ============================================================

def get_fiedler_vector(
        eigvals,
        eigvecs):

    if len(eigvals) < 2:

        raise ValueError(
            "At least two eigenvalues are required."
        )

    return eigvecs[:, 1]


# ============================================================
# 7. FIEDLER ORIENTATION
# ============================================================

def orient_fiedler_vector(fiedler):

    fiedler = fiedler.copy()

    if fiedler[0] > 0:

        fiedler *= -1

    return fiedler


# ============================================================
# 8. PSEUDOINVERSE
# ============================================================

def compute_pseudoinverse(
        L,
        tolerance=1e-10):

    eigvals, eigvecs = np.linalg.eigh(L)

    L_dagger = np.zeros_like(
        L,
        dtype=float
    )

    for k, lam in enumerate(eigvals):

        if lam > tolerance:

            vk = eigvecs[:, k]

            L_dagger += (
                1.0 / lam
            ) * np.outer(
                vk,
                vk
            )

    return L_dagger


# ============================================================
# 9. EFFECTIVE RESISTANCE
# ============================================================

def compute_effective_resistance(L):

    """
    Compute the full pairwise effective-resistance matrix.
    """

    L_dagger = compute_pseudoinverse(L)

    n = L.shape[0]

    resistance = np.zeros(
        (n, n)
    )

    # More compact and numerically efficient formulation:
    #
    # R_ij =
    # Ldagger_ii + Ldagger_jj - 2 Ldagger_ij

    diagonal = np.diag(
        L_dagger
    )

    resistance = (
        diagonal[:, None]
        +
        diagonal[None, :]
        -
        2.0 * L_dagger
    )

    # Remove tiny numerical negative values.
    resistance[
        np.abs(resistance) < 1e-12
    ] = 0.0

    return resistance


# ============================================================
# 10. FIEDLER EDGE SENSITIVITY
# ============================================================

def fiedler_edge_sensitivities(
        G,
        fiedler):

    sensitivities = {}

    for u, v in G.edges():

        sensitivities[(u, v)] = (
            fiedler[u] -
            fiedler[v]
        ) ** 2

    return sensitivities


# ============================================================
# 11. CONSTRUCT COARSE MATRIX
# ============================================================

def construct_coarse_matrix(
        clusters):

    n = sum(
        len(cluster)
        for cluster in clusters
    )

    m = len(clusters)

    P = np.zeros(
        (n, m)
    )

    for a, cluster in enumerate(
            clusters):

        if len(cluster) == 0:

            raise ValueError(
                "Empty cluster encountered."
            )

        normalization = (
            1.0 /
            np.sqrt(len(cluster))
        )

        for node in cluster:

            P[node, a] = normalization

    return P


# ============================================================
# 12. COARSE GRAIN LAPLACIAN
# ============================================================

def coarse_grain_laplacian(
        L,
        P):

    return P.T @ L @ P


# ============================================================
# 13. LAPLACIAN -> GRAPH
# ============================================================

def laplacian_to_graph(L):

    G = nx.Graph()

    n = L.shape[0]

    G.add_nodes_from(
        range(n)
    )

    for i in range(n):

        for j in range(i + 1, n):

            weight = -L[i, j]

            if weight > 1e-12:

                G.add_edge(
                    i,
                    j,
                    weight=weight
                )

    return G


# ============================================================
# 14. LINEAR KURAMOTO RESPONSE
# ============================================================

def compute_linear_response(
        L,
        omega,
        K=1.0):

    n = L.shape[0]

    projector = (
        np.eye(n)
        -
        np.ones((n, n)) / n
    )

    omega_perp = (
        projector @ omega
    )

    L_dagger = compute_pseudoinverse(
        L
    )

    theta = (
        1.0 / K
    ) * L_dagger @ omega_perp

    return theta


# ============================================================
# 15. SPECTRAL CLUSTERING
# ============================================================

def spectral_cluster_nodes(
        L,
        n_clusters):

    """
    Spectral clustering using the lowest nontrivial
    Laplacian eigenvectors.
    """

    eigvals, eigvecs = (
        spectral_decomposition(L)
    )

    if n_clusters == 2:

        fiedler = orient_fiedler_vector(
            eigvecs[:, 1]
        )

        cluster_a = [
            i
            for i, value in enumerate(fiedler)
            if value <= 0
        ]

        cluster_b = [
            i
            for i, value in enumerate(fiedler)
            if value > 0
        ]

        if (
            len(cluster_a) == 0
            or
            len(cluster_b) == 0
        ):

            return spectral_kmeans(
                eigvecs[:, 1:2],
                n_clusters
            )

        return [
            cluster_a,
            cluster_b
        ]

    embedding = eigvecs[
        :,
        1:n_clusters
    ]

    return spectral_kmeans(
        embedding,
        n_clusters
    )


# ============================================================
# 16. SIMPLE DETERMINISTIC K-MEANS
# ============================================================

def spectral_kmeans(
        data,
        n_clusters,
        max_iter=100):

    n_samples = data.shape[0]

    if n_samples < n_clusters:

        raise ValueError(
            "Number of samples must be >= "
            "number of clusters."
        )

    indices = np.linspace(
        0,
        n_samples - 1,
        n_clusters,
        dtype=int
    )

    centres = data[
        indices
    ].copy()

    labels = np.zeros(
        n_samples,
        dtype=int
    )

    for _ in range(max_iter):

        distances = np.zeros(
            (n_samples, n_clusters)
        )

        for k in range(n_clusters):

            distances[:, k] = np.linalg.norm(
                data - centres[k],
                axis=1
            )

        new_labels = np.argmin(
            distances,
            axis=1
        )

        if np.array_equal(
                labels,
                new_labels):

            break

        labels = new_labels

        for k in range(n_clusters):

            members = data[
                labels == k
            ]

            if len(members) > 0:

                centres[k] = (
                    members.mean(axis=0)
                )

    clusters = []

    for k in range(n_clusters):

        cluster = list(
            np.where(
                labels == k
            )[0]
        )

        clusters.append(
            cluster
        )

    clusters.sort(
        key=lambda c: min(c)
    )

    return clusters


# ============================================================
# 17. GRAPH VISUALISATION
# ============================================================

def visualize_graph(
        G,
        pos=None,
        node_colors=None,
        title="Graph",
        subtitle=None,
        edge_labels=True,
        figsize=(10, 7),
        filename=None):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    if pos is None:

        pos = nx.spring_layout(
            G,
            seed=42
        )

    if node_colors is None:

        node_colors = [
            "lightblue"
            for _ in G.nodes()
        ]

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=850,
        edgecolors="black",
        linewidths=1.2
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        width=2.0
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=11,
        font_weight="bold"
    )

    if edge_labels:

        labels = nx.get_edge_attributes(
            G,
            "weight"
        )

        labels = {
            edge: f"{weight:.2f}"
            for edge, weight in labels.items()
        }

        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels=labels,
            font_size=8
        )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    if subtitle is not None:

        ax.text(
            0.5,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            color="dimgray"
        )

    ax.axis("off")

    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    if filename is not None:

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)


# ============================================================
# 18. FIEDLER VISUALISATION
# ============================================================

def visualize_fiedler(
        fiedler,
        title="Fiedler Vector",
        subtitle=None,
        filename=None):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    colors = [
        "royalblue" if x < 0
        else "crimson"
        for x in fiedler
    ]

    ax.bar(
        range(len(fiedler)),
        fiedler,
        color=colors,
        edgecolor="black"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=1
    )

    ax.set_xlabel(
        "Node",
        fontsize=12
    )

    ax.set_ylabel(
        "Fiedler component",
        fontsize=12
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    if subtitle is not None:

        ax.text(
            0.5,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            color="dimgray"
        )

    ax.set_xticks(
        range(len(fiedler))
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    if filename is not None:

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)


# ============================================================
# 19. RESISTANCE VISUALISATION
# ============================================================

def visualize_resistance(
        resistance,
        title="Effective Resistance Matrix",
        subtitle=None,
        filename=None):

    """
    Visualise one effective-resistance matrix.

    The colourbar is explicitly attached to the axis so that
    it remains outside the heatmap.
    """

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        resistance,
        cmap="magma",
        interpolation="nearest",
        aspect="equal"
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label(
        "Effective resistance",
        fontsize=11
    )

    ax.set_xlabel(
        "Node",
        fontsize=12
    )

    ax.set_ylabel(
        "Node",
        fontsize=12
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    if subtitle is not None:

        ax.text(
            0.5,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            color="dimgray"
        )

    n = resistance.shape[0]

    tick_step = max(
        1,
        int(np.ceil(n / 12))
    )

    ticks = np.arange(
        0,
        n,
        tick_step
    )

    ax.set_xticks(
        ticks
    )

    ax.set_yticks(
        ticks
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    if filename is not None:

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)


# ============================================================
# 19B. CLUSTER-AVERAGED MICROSCOPIC RESISTANCE
# ============================================================

def compute_cluster_averaged_resistance(
        resistance,
        clusters):

    """
    Coarse-grain the microscopic resistance matrix by averaging
    resistance over each pair of microscopic clusters.

    Rbar^(0)_ab =
        1 / (|C_a||C_b|)
        sum_{i in C_a}
        sum_{j in C_b}
        R^(0)_ij

    The diagonal blocks therefore contain the mean pairwise
    resistance within each microscopic cluster.
    """

    n_clusters = len(
        clusters
    )

    coarse_resistance = np.zeros(
        (n_clusters, n_clusters),
        dtype=float
    )

    for a, cluster_a in enumerate(
            clusters):

        for b, cluster_b in enumerate(
                clusters):

            block = resistance[
                np.ix_(
                    cluster_a,
                    cluster_b
                )
            ]

            coarse_resistance[a, b] = (
                np.mean(block)
            )

    return coarse_resistance


# ============================================================
# 19C. RESISTANCE COMPARISON VISUALISATION
# ============================================================

def visualize_resistance_comparison(
        resistance,
        resistance1,
        clusters,
        filename=None):

    """
    Compare microscopic and renormalised effective resistance.

    Panel 1:
        R^(0), microscopic effective resistance.

    Panel 2:
        R^(1), renormalised effective resistance.

    Panel 3:
        Delta R =
            R^(1) - Rbar^(0)

    where Rbar^(0) is the cluster-averaged microscopic
    resistance.

    All heatmaps have independent colourbars positioned in
    dedicated spaces outside the plots.
    """

    # --------------------------------------------------------
    # Cluster-averaged microscopic resistance
    # --------------------------------------------------------

    resistance0_coarse = (
        compute_cluster_averaged_resistance(
            resistance,
            clusters
        )
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    difference = (
        resistance1
        -
        resistance0_coarse
    )

    # --------------------------------------------------------
    # Shared colour scale for scale 0 and scale 1
    # --------------------------------------------------------

    resistance_min = min(
        np.min(resistance),
        np.min(resistance1)
    )

    resistance_max = max(
        np.max(resistance),
        np.max(resistance1)
    )

    # --------------------------------------------------------
    # Symmetric scale for difference
    # --------------------------------------------------------

    difference_max = np.max(
        np.abs(difference)
    )

    if difference_max < 1e-14:

        difference_max = 1.0

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(18, 7)
    )

    gs = fig.add_gridspec(
        1,
        6,
        width_ratios=[
            1.0,
            0.055,
            0.65,
            0.055,
            0.65,
            0.055
        ],
        wspace=0.45
    )

    ax0 = fig.add_subplot(
        gs[0, 0]
    )

    cax0 = fig.add_subplot(
        gs[0, 1]
    )

    ax1 = fig.add_subplot(
        gs[0, 2]
    )

    cax1 = fig.add_subplot(
        gs[0, 3]
    )

    axdiff = fig.add_subplot(
        gs[0, 4]
    )

    caxdiff = fig.add_subplot(
        gs[0, 5]
    )

    # ========================================================
    # SCALE 0
    # ========================================================

    im0 = ax0.imshow(
        resistance,
        cmap="magma",
        interpolation="nearest",
        vmin=resistance_min,
        vmax=resistance_max,
        aspect="equal"
    )

    ax0.set_title(
        "Scale 0\nMicroscopic Resistance",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    ax0.set_xlabel(
        "Microscopic node",
        fontsize=11
    )

    ax0.set_ylabel(
        "Microscopic node",
        fontsize=11
    )

    n0 = resistance.shape[0]

    tick_step0 = max(
        1,
        int(np.ceil(n0 / 10))
    )

    ticks0 = np.arange(
        0,
        n0,
        tick_step0
    )

    ax0.set_xticks(
        ticks0
    )

    ax0.set_yticks(
        ticks0
    )

    ax0.tick_params(
        axis="both",
        labelsize=9
    )

    # --------------------------------------------------------
    # Scale 0 colourbar
    # --------------------------------------------------------

    cbar0 = fig.colorbar(
        im0,
        cax=cax0
    )

    cbar0.set_label(
        "Effective resistance",
        fontsize=9
    )

    cbar0.ax.tick_params(
        labelsize=8
    )

    # ========================================================
    # SCALE 1
    # ========================================================

    im1 = ax1.imshow(
        resistance1,
        cmap="magma",
        interpolation="nearest",
        vmin=resistance_min,
        vmax=resistance_max,
        aspect="equal"
    )

    ax1.set_title(
        "Scale 1\nRenormalised Resistance",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    ax1.set_xlabel(
        "Cluster",
        fontsize=11
    )

    ax1.set_ylabel(
        "Cluster",
        fontsize=11
    )

    n1 = resistance1.shape[0]

    ticks1 = np.arange(
        n1
    )

    ax1.set_xticks(
        ticks1
    )

    ax1.set_yticks(
        ticks1
    )

    ax1.set_xticklabels(
        [
            f"$C_{{{i + 1}}}$"
            for i in range(n1)
        ],
        fontsize=9
    )

    ax1.set_yticklabels(
        [
            f"$C_{{{i + 1}}}$"
            for i in range(n1)
        ],
        fontsize=9
    )

    # --------------------------------------------------------
    # Scale 1 colourbar
    # --------------------------------------------------------

    cbar1 = fig.colorbar(
        im1,
        cax=cax1
    )

    cbar1.set_label(
        "Effective resistance",
        fontsize=9
    )

    cbar1.ax.tick_params(
        labelsize=8
    )

    # ========================================================
    # DIFFERENCE
    # ========================================================

    imdiff = axdiff.imshow(
        difference,
        cmap="coolwarm",
        interpolation="nearest",
        vmin=-difference_max,
        vmax=difference_max,
        aspect="equal"
    )

    axdiff.set_title(
        "RG Difference\n"
        r"$R^{(1)}-\overline{R}^{(0)}$",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    axdiff.set_xlabel(
        "Cluster",
        fontsize=11
    )

    axdiff.set_ylabel(
        "Cluster",
        fontsize=11
    )

    axdiff.set_xticks(
        ticks1
    )

    axdiff.set_yticks(
        ticks1
    )

    axdiff.set_xticklabels(
        [
            f"$C_{{{i + 1}}}$"
            for i in range(n1)
        ],
        fontsize=9
    )

    axdiff.set_yticklabels(
        [
            f"$C_{{{i + 1}}}$"
            for i in range(n1)
        ],
        fontsize=9
    )

    # --------------------------------------------------------
    # Difference colourbar
    # --------------------------------------------------------

    cbar_diff = fig.colorbar(
        imdiff,
        cax=caxdiff
    )

    cbar_diff.set_label(
        r"$\Delta R$",
        fontsize=9
    )

    cbar_diff.ax.tick_params(
        labelsize=8
    )

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        "Effective Resistance Across the Spectral RG Flow",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    fig.text(
        0.5,
        0.935,
        "Microscopic resistance, renormalised resistance, "
        "and cluster-level RG difference",
        ha="center",
        fontsize=11,
        color="dimgray"
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.subplots_adjust(
        left=0.055,
        right=0.98,
        bottom=0.12,
        top=0.84,
        wspace=0.48
    )

    if filename is not None:

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)


# ============================================================
# 20. RG FLOW VISUALISATION
# ============================================================

def visualize_rg_flow(
        G_micro,
        G_coarse,
        clusters,
        filename=None):

    """
    Visualise the microscopic-to-coarse RG transformation.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7)
    )

    # --------------------------------------------------------
    # Microscopic graph
    # --------------------------------------------------------

    pos_micro = nx.spring_layout(
        G_micro,
        seed=42,
        k=1.5
    )

    cluster_palette = [
        "royalblue",
        "crimson",
        "seagreen",
        "darkorange",
        "purple",
        "goldenrod",
        "deeppink",
        "teal"
    ]

    node_to_cluster = {}

    for cluster_id, cluster in enumerate(
            clusters):

        for node in cluster:

            node_to_cluster[node] = cluster_id

    colors_micro = []

    for node in G_micro.nodes():

        cluster_id = node_to_cluster.get(
            node,
            0
        )

        colors_micro.append(
            cluster_palette[
                cluster_id %
                len(cluster_palette)
            ]
        )

    nx.draw_networkx(
        G_micro,
        pos_micro,
        ax=axes[0],
        node_color=colors_micro,
        node_size=750,
        edge_color="gray",
        width=1.8,
        with_labels=True,
        font_weight="bold"
    )

    axes[0].set_title(
        "Scale 0\nMicroscopic QFERN Network",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Coarse graph
    # --------------------------------------------------------

    pos_coarse = nx.spring_layout(
        G_coarse,
        seed=42
    )

    coarse_colors = [
        cluster_palette[
            i % len(cluster_palette)
        ]
        for i in G_coarse.nodes()
    ]

    labels = {
        i: f"C{i + 1}"
        for i in G_coarse.nodes()
    }

    nx.draw_networkx(
        G_coarse,
        pos_coarse,
        ax=axes[1],
        node_color=coarse_colors,
        node_size=1500,
        edge_color="black",
        width=3,
        with_labels=True,
        labels=labels,
        font_weight="bold",
        font_size=12
    )

    axes[1].set_title(
        "Scale 1\nSpectral Coarse-Grained Network",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        "QFERN → Spectral RG Coarse-Graining",
        fontsize=19,
        fontweight="bold",
        y=0.98
    )

    fig.text(
        0.5,
        0.935,
        "Microscopic network transformed into collective "
        "spectral variables",
        ha="center",
        fontsize=11,
        color="dimgray"
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.91]
    )

    if filename is not None:

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)


# ============================================================
# 21. EIGENVALUE SPECTRUM
# ============================================================

def visualize_spectrum(
        eigvals,
        filename=None):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        range(len(eigvals)),
        eigvals,
        "o-",
        color="navy",
        linewidth=2
    )

    ax.set_xlabel(
        "Eigenvalue index",
        fontsize=12
    )

    ax.set_ylabel(
        "Laplacian eigenvalue",
        fontsize=12
    )

    ax.set_title(
        "Scale 0 — Laplacian Spectrum",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    if filename is not None:

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)


# ============================================================
# 22. QFERN EDGE SENSITIVITY
# ============================================================

def visualize_sensitivities(
        sensitivities,
        filename=None):

    edges = list(
        sensitivities.keys()
    )

    values = [
        sensitivities[e]
        for e in edges
    ]

    labels = [
        f"{u}-{v}"
        for u, v in edges
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        range(len(values)),
        values,
        color="darkorange",
        edgecolor="black"
    )

    ax.set_xlabel(
        "Edge",
        fontsize=12
    )

    ax.set_ylabel(
        r"$(f_i-f_j)^2$",
        fontsize=12
    )

    ax.set_title(
        "Scale 0 — QFERN Fiedler Edge Sensitivity",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.set_xticks(
        range(len(labels))
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    if filename is not None:

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)


# ============================================================
# 23. KURAMOTO RESPONSE VISUALISATION
# ============================================================

def visualize_response(
        theta,
        filename=None):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    colors = [
        "royalblue" if x < 0
        else "crimson"
        for x in theta
    ]

    ax.bar(
        range(len(theta)),
        theta,
        color=colors,
        edgecolor="black"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=1
    )

    ax.set_xlabel(
        "Node",
        fontsize=12
    )

    ax.set_ylabel(
        r"$\theta_i$",
        fontsize=12
    )

    ax.set_title(
        "Scale 0 — Linearised Kuramoto Phase Response",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    if filename is not None:

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close(fig)


# ============================================================
# 24. SAVE NUMERICAL DATA
# ============================================================

def save_numerical_data(
        output_dir,
        L,
        L1,
        fiedler,
        fiedler1,
        resistance,
        resistance1,
        theta,
        theta1,
        eigvals,
        eigvals1,
        P,
        resistance0_coarse=None,
        resistance_difference=None):

    filename = os.path.join(
        output_dir,
        "qfern_rg_data.npz"
    )

    data = {
        "L": L,
        "L1": L1,
        "fiedler": fiedler,
        "fiedler1": fiedler1,
        "resistance": resistance,
        "resistance1": resistance1,
        "theta": theta,
        "theta1": theta1,
        "eigvals": eigvals,
        "eigvals1": eigvals1,
        "P": P
    }

    if resistance0_coarse is not None:

        data[
            "resistance0_coarse"
        ] = resistance0_coarse

    if resistance_difference is not None:

        data[
            "resistance_difference"
        ] = resistance_difference

    np.savez(
        filename,
        **data
    )


# ============================================================
# 25. SAVE TEXT SUMMARY
# ============================================================

def save_summary(
        output_dir,
        G,
        clusters,
        topologies,
        bridge_type,
        eigvals,
        eigvals1,
        lambda2,
        lambda2_1,
        fiedler_error,
        response_error,
        resistance=None,
        resistance1=None,
        resistance_difference=None):

    filename = os.path.join(
        output_dir,
        "simulation_summary.txt"
    )

    with open(
            filename,
            "w",
            encoding="utf-8") as f:

        f.write(
            "QFERN + SPECTRAL RG SIMULATION\n"
        )

        f.write(
            "================================\n\n"
        )

        f.write(
            f"Number of clusters: "
            f"{len(clusters)}\n"
        )

        f.write(
            f"Cluster topologies: "
            f"{topologies}\n"
        )

        f.write(
            f"Bridge topology: "
            f"{bridge_type}\n"
        )

        f.write(
            f"Total nodes: "
            f"{G.number_of_nodes()}\n"
        )

        f.write(
            f"Total edges: "
            f"{G.number_of_edges()}\n\n"
        )

        f.write(
            "Clusters:\n"
        )

        for i, cluster in enumerate(
                clusters):

            f.write(
                f"  C{i + 1}: {cluster}\n"
            )

        f.write(
            "\nMicroscopic eigenvalues:\n"
        )

        f.write(
            str(
                np.round(
                    eigvals,
                    8
                )
            )
        )

        f.write(
            "\n\nCoarse eigenvalues:\n"
        )

        f.write(
            str(
                np.round(
                    eigvals1,
                    8
                )
            )
        )

        f.write(
            "\n\nMicroscopic lambda_2: "
            f"{lambda2:.10f}\n"
        )

        f.write(
            "Coarse lambda_2: "
            f"{lambda2_1:.10f}\n"
        )

        f.write(
            "\nFiedler lifting error: "
            f"{fiedler_error:.10e}\n"
        )

        f.write(
            "Response lifting error: "
            f"{response_error:.10e}\n"
        )

        # ----------------------------------------------------
        # Resistance summary
        # ----------------------------------------------------

        if resistance is not None:

            f.write(
                "\nResistance statistics — Scale 0:\n"
            )

            f.write(
                f"  Maximum: "
                f"{np.max(resistance):.10f}\n"
            )

            f.write(
                f"  Mean: "
                f"{np.mean(resistance):.10f}\n"
            )

        if resistance1 is not None:

            f.write(
                "\nResistance statistics — Scale 1:\n"
            )

            f.write(
                f"  Maximum: "
                f"{np.max(resistance1):.10f}\n"
            )

            f.write(
                f"  Mean: "
                f"{np.mean(resistance1):.10f}\n"
            )

        if resistance_difference is not None:

            f.write(
                "\nRG resistance difference:\n"
            )

            f.write(
                "  Delta R = "
                "R^(1) - Rbar^(0)\n"
            )

            f.write(
                f"  Maximum absolute difference: "
                f"{np.max(np.abs(resistance_difference)):.10f}\n"
            )

            f.write(
                f"  Mean absolute difference: "
                f"{np.mean(np.abs(resistance_difference)):.10f}\n"
            )


# ============================================================
# 26. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 75)

    print(
        "QFERN + SPECTRAL RENORMALISATION GROUP"
    )

    print(
        "General Multi-Cluster Simulation"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # User input: number of clusters
    # --------------------------------------------------------

    while True:

        try:

            n_clusters = int(
                input(
                    "\nHow many clusters would "
                    "you like to model? "
                )
            )

            if n_clusters >= 2:
                break

            print(
                "Please enter at least 2 clusters."
            )

        except ValueError:

            print(
                "Please enter an integer."
            )

    # --------------------------------------------------------
    # User input: cluster sizes
    # --------------------------------------------------------

    cluster_sizes = []

    print(
        "\nEnter the number of nodes "
        "for each cluster."
    )

    for i in range(n_clusters):

        while True:

            try:

                size = int(
                    input(
                        f"Nodes in cluster {i + 1}: "
                    )
                )

                if size >= 2:
                    break

                print(
                    "Cluster must contain "
                    "at least 2 nodes."
                )

            except ValueError:

                print(
                    "Please enter an integer."
                )

        cluster_sizes.append(
            size
        )

    # --------------------------------------------------------
    # User input: cluster topology
    # --------------------------------------------------------

    valid_topologies = [
        "strogatz",
        "barabasi_albert",
        "karate_club",
        "star",
        "ring",
        "line"
    ]

    print(
        "\nAvailable cluster topologies:"
    )

    for topology in valid_topologies:

        print(
            f"  - {topology}"
        )

    topologies = []

    for i, size in enumerate(
            cluster_sizes):

        while True:

            topology = input(
                f"\nTopology for cluster "
                f"{i + 1} ({size} nodes): "
            ).lower().strip()

            if topology in valid_topologies:

                if (
                    topology == "karate_club"
                    and size != 34
                ):

                    print(
                        "Karate Club requires "
                        "exactly 34 nodes."
                    )

                    continue

                if (
                    topology == "ring"
                    and size < 3
                ):

                    print(
                        "Ring requires at least "
                        "3 nodes."
                    )

                    continue

                topologies.append(
                    topology
                )

                break

            print(
                "Unknown topology. "
                "Please choose one of:"
            )

            print(
                ", ".join(
                    valid_topologies
                )
            )

    # --------------------------------------------------------
    # User input: bridge structure
    # --------------------------------------------------------

    print(
        "\nHow should the clusters be connected?"
    )

    print(
        "  chain = C1--C2--C3--..."
    )

    print(
        "  ring  = C1--C2--...--Cn--C1"
    )

    print(
        "  all   = every cluster connected"
    )

    while True:

        bridge_type = input(
            "\nBridge type [chain/ring/all]: "
        ).lower().strip()

        if bridge_type in [
            "chain",
            "ring",
            "all"
        ]:

            break

        print(
            "Please enter chain, ring, or all."
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = os.path.join(
        RESULTS_ROOT,
        "simulation"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print(
        "\nResults will be saved to:"
    )

    print(
        os.path.abspath(
            output_dir
        )
    )

    # ========================================================
    # CONSTRUCT MICROSCOPIC GRAPH
    # ========================================================

    G, original_clusters = (
        create_clustered_network(
            cluster_sizes,
            topologies,
            bridge_type=bridge_type,
            seed=DEFAULT_SEED
        )
    )

    print(
        "\nMicroscopic network created."
    )

    print(
        f"Nodes: {G.number_of_nodes()}"
    )

    print(
        f"Edges: {G.number_of_edges()}"
    )

    # ========================================================
    # LAPLACIAN
    # ========================================================

    L = compute_laplacian(
        G
    )

    print(
        "\nMicroscopic Laplacian L^(0):"
    )

    print(
        np.round(
            L,
            4
        )
    )

    # ========================================================
    # SPECTRUM
    # ========================================================

    eigvals, eigvecs = (
        spectral_decomposition(
            L
        )
    )

    lambda2 = eigvals[1]

    print(
        "\nMicroscopic eigenvalues:"
    )

    print(
        np.round(
            eigvals,
            8
        )
    )

    print(
        f"\nMicroscopic lambda_2 = "
        f"{lambda2:.8f}"
    )

    # ========================================================
    # FIEDLER VECTOR
    # ========================================================

    fiedler = get_fiedler_vector(
        eigvals,
        eigvecs
    )

    fiedler = orient_fiedler_vector(
        fiedler
    )

    print(
        "\nFiedler vector:"
    )

    print(
        np.round(
            fiedler,
            8
        )
    )

    # ========================================================
    # SCALE-0 EFFECTIVE RESISTANCE
    # ========================================================

    resistance = (
        compute_effective_resistance(
            L
        )
    )

    print(
        "\nEffective resistance matrix "
        "R^(0):"
    )

    print(
        np.round(
            resistance,
            6
        )
    )

    # ========================================================
    # QFERN EDGE SENSITIVITY
    # ========================================================

    sensitivities = (
        fiedler_edge_sensitivities(
            G,
            fiedler
        )
    )

    print(
        "\nQFERN edge sensitivities:"
    )

    for edge, value in sensitivities.items():

        print(
            f"Edge {edge}: "
            f"{value:.8f}"
        )

    # ========================================================
    # SPECTRAL CLUSTERING
    # ========================================================

    clusters = (
        spectral_cluster_nodes(
            L,
            n_clusters
        )
    )

    print(
        "\nSpectral coarse-graining clusters:"
    )

    for i, cluster in enumerate(
            clusters):

        print(
            f"Cluster {i + 1}: "
            f"{cluster}"
        )

    # ========================================================
    # COARSE-GRAINING MATRIX
    # ========================================================

    P = construct_coarse_matrix(
        clusters
    )

    print(
        "\nCoarse-graining matrix P:"
    )

    print(
        np.round(
            P,
            6
        )
    )

    print(
        "\nP^T P:"
    )

    print(
        np.round(
            P.T @ P,
            6
        )
    )

    # ========================================================
    # COARSE LAPLACIAN
    # ========================================================

    L1 = coarse_grain_laplacian(
        L,
        P
    )

    print(
        "\nCoarse Laplacian L^(1):"
    )

    print(
        np.round(
            L1,
            8
        )
    )

    # ========================================================
    # COARSE SPECTRUM
    # ========================================================

    eigvals1, eigvecs1 = (
        spectral_decomposition(
            L1
        )
    )

    print(
        "\nCoarse eigenvalues:"
    )

    print(
        np.round(
            eigvals1,
            8
        )
    )

    lambda2_1 = (
        eigvals1[1]
        if len(eigvals1) > 1
        else np.nan
    )

    print(
        f"\nCoarse lambda_2 = "
        f"{lambda2_1:.8f}"
    )

    # ========================================================
    # COARSE FIEDLER
    # ========================================================

    fiedler1 = get_fiedler_vector(
        eigvals1,
        eigvecs1
    )

    fiedler1 = orient_fiedler_vector(
        fiedler1
    )

    # ========================================================
    # LIFT COARSE FIEDLER
    # ========================================================

    lifted_fiedler = (
        P @ fiedler1
    )

    if np.dot(
            lifted_fiedler,
            fiedler) < 0:

        lifted_fiedler *= -1

    fiedler_error = np.linalg.norm(
        fiedler -
        lifted_fiedler
    )

    print(
        "\nFiedler lifting error:"
    )

    print(
        f"{fiedler_error:.10e}"
    )

    # ========================================================
    # COARSE PSEUDOINVERSE
    # ========================================================

    L1_dagger = (
        compute_pseudoinverse(
            L1
        )
    )

    print(
        "\nCoarse pseudoinverse:"
    )

    print(
        np.round(
            L1_dagger,
            8
        )
    )

    # ========================================================
    # SCALE-1 EFFECTIVE RESISTANCE
    # ========================================================

    resistance1 = (
        compute_effective_resistance(
            L1
        )
    )

    print(
        "\nRenormalised effective "
        "resistance matrix R^(1):"
    )

    print(
        np.round(
            resistance1,
            8
        )
    )

    # ========================================================
    # CLUSTER-AVERAGED SCALE-0 RESISTANCE
    # ========================================================

    resistance0_coarse = (
        compute_cluster_averaged_resistance(
            resistance,
            clusters
        )
    )

    print(
        "\nCluster-averaged microscopic "
        "resistance:"
    )

    print(
        np.round(
            resistance0_coarse,
            8
        )
    )

    # ========================================================
    # RESISTANCE DIFFERENCE
    # ========================================================

    resistance_difference = (
        resistance1
        -
        resistance0_coarse
    )

    print(
        "\nResistance RG difference:"
    )

    print(
        np.round(
            resistance_difference,
            8
        )
    )

    # ========================================================
    # KURAMOTO FORCING
    # ========================================================

    omega = fiedler.copy()

    theta = (
        compute_linear_response(
            L,
            omega,
            K=1.0
        )
    )

    omega1 = (
        P.T @ omega
    )

    theta1 = (
        compute_linear_response(
            L1,
            omega1,
            K=1.0
        )
    )

    theta_lifted = (
        P @ theta1
    )

    response_error = np.linalg.norm(
        theta -
        theta_lifted
    )

    print(
        "\nMicroscopic response:"
    )

    print(
        np.round(
            theta,
            8
        )
    )

    print(
        "\nCoarse response:"
    )

    print(
        np.round(
            theta1,
            8
        )
    )

    print(
        "\nResponse lifting error:"
    )

    print(
        f"{response_error:.10e}"
    )

    # ========================================================
    # COARSE GRAPH
    # ========================================================

    G1 = laplacian_to_graph(
        L1
    )

    # ========================================================
    # VISUALISATIONS
    # ========================================================

    palette = [
        "royalblue",
        "crimson",
        "seagreen",
        "darkorange",
        "purple",
        "goldenrod",
        "deeppink",
        "teal"
    ]

    node_colors = []

    node_to_cluster = {}

    for i, cluster in enumerate(
            clusters):

        for node in cluster:

            node_to_cluster[node] = i

    for node in G.nodes():

        cluster_id = (
            node_to_cluster[node]
        )

        node_colors.append(
            palette[
                cluster_id %
                len(palette)
            ]
        )

    # ========================================================
    # GRAPH LAYOUT
    # ========================================================

    pos_micro = nx.spring_layout(
        G,
        seed=42,
        k=1.5
    )

    # ========================================================
    # 01 MICRO GRAPH
    # ========================================================

    visualize_graph(
        G,
        pos=pos_micro,
        node_colors=node_colors,
        title="Scale 0 — Microscopic QFERN Network",
        subtitle=(
            f"{n_clusters} spectral clusters  |  "
            f"{G.number_of_nodes()} nodes  |  "
            f"{G.number_of_edges()} edges"
        ),
        filename=os.path.join(
            output_dir,
            "01_micro_graph.png"
        )
    )

    # ========================================================
    # 02 FIEDLER
    # ========================================================

    visualize_fiedler(
        fiedler,
        title="Scale 0 — Fiedler Vector",
        subtitle=(
            "Lowest non-trivial Laplacian mode "
            "identifying large-scale connectivity"
        ),
        filename=os.path.join(
            output_dir,
            "02_fiedler_vector.png"
        )
    )

    # ========================================================
    # 03 SCALE-0 RESISTANCE
    # ========================================================

    visualize_resistance(
        resistance,
        title="Scale 0 — Effective Resistance Matrix",
        subtitle=(
            "Pairwise network response derived "
            "from the Moore–Penrose pseudoinverse"
        ),
        filename=os.path.join(
            output_dir,
            "03_effective_resistance.png"
        )
    )

    # ========================================================
    # 03B SCALE-1 RESISTANCE
    # ========================================================

    visualize_resistance(
        resistance1,
        title=(
            "Scale 1 — Renormalised "
            "Effective Resistance Matrix"
        ),
        subtitle=(
            "Effective resistance of the "
            "spectral coarse-grained network"
        ),
        filename=os.path.join(
            output_dir,
            "03b_renormalised_effective_resistance.png"
        )
    )

    # ========================================================
    # 03C RESISTANCE COMPARISON
    # ========================================================

    visualize_resistance_comparison(
        resistance,
        resistance1,
        clusters,
        filename=os.path.join(
            output_dir,
            "03c_effective_resistance_comparison.png"
        )
    )

    # ========================================================
    # 04 COARSE GRAPH
    # ========================================================

    coarse_colors = [
        palette[
            i % len(palette)
        ]
        for i in G1.nodes()
    ]

    visualize_graph(
        G1,
        node_colors=coarse_colors,
        title="Scale 1 — Coarse-Grained QFERN Network",
        subtitle=(
            f"{n_clusters} collective spectral variables"
        ),
        filename=os.path.join(
            output_dir,
            "04_coarse_graph.png"
        )
    )

    # ========================================================
    # 05 COARSE FIEDLER
    # ========================================================

    visualize_fiedler(
        fiedler1,
        title="Scale 1 — Coarse Fiedler Vector",
        subtitle=(
            "Fiedler structure after spectral "
            "coarse-graining"
        ),
        filename=os.path.join(
            output_dir,
            "05_coarse_fiedler_vector.png"
        )
    )

    # ========================================================
    # 06 RG FLOW
    # ========================================================

    visualize_rg_flow(
        G,
        G1,
        clusters,
        filename=os.path.join(
            output_dir,
            "06_rg_flow.png"
        )
    )

    # ========================================================
    # 07 EIGENVALUE SPECTRUM
    # ========================================================

    visualize_spectrum(
        eigvals,
        filename=os.path.join(
            output_dir,
            "07_eigenvalue_spectrum.png"
        )
    )

    # ========================================================
    # 08 QFERN EDGE SENSITIVITY
    # ========================================================

    visualize_sensitivities(
        sensitivities,
        filename=os.path.join(
            output_dir,
            "08_qfern_edge_sensitivity.png"
        )
    )

    # ========================================================
    # 09 KURAMOTO RESPONSE
    # ========================================================

    visualize_response(
        theta,
        filename=os.path.join(
            output_dir,
            "09_kuramoto_response.png"
        )
    )

    # ========================================================
    # SAVE NUMERICAL DATA
    # ========================================================

    save_numerical_data(
        output_dir,
        L,
        L1,
        fiedler,
        fiedler1,
        resistance,
        resistance1,
        theta,
        theta1,
        eigvals,
        eigvals1,
        P,
        resistance0_coarse,
        resistance_difference
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    save_summary(
        output_dir,
        G,
        clusters,
        topologies,
        bridge_type,
        eigvals,
        eigvals1,
        lambda2,
        lambda2_1,
        fiedler_error,
        response_error,
        resistance,
        resistance1,
        resistance_difference
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")

    print("=" * 75)

    print(
        "QFERN + RG SIMULATION COMPLETE"
    )

    print("=" * 75)

    print(
        f"\nNumber of clusters: "
        f"{n_clusters}"
    )

    print(
        f"Cluster sizes: "
        f"{cluster_sizes}"
    )

    print(
        f"Cluster topologies: "
        f"{topologies}"
    )

    print(
        f"Inter-cluster topology: "
        f"{bridge_type}"
    )

    print(
        f"\nλ₂^(0) = "
        f"{lambda2:.8f}"
    )

    print(
        f"λ₂^(1) = "
        f"{lambda2_1:.8f}"
    )

    print(
        "\nFiedler lifting error = "
        f"{fiedler_error:.6e}"
    )

    print(
        "Response lifting error = "
        f"{response_error:.6e}"
    )

    print(
        "\nResistance RG difference:"
    )

    print(
        f"Maximum |ΔR| = "
        f"{np.max(np.abs(resistance_difference)):.6e}"
    )

    print(
        f"Mean |ΔR| = "
        f"{np.mean(np.abs(resistance_difference)):.6e}"
    )

    print(
        "\nPlots saved to:"
    )

    print(
        os.path.abspath(
            output_dir
        )
    )

    print(
        "\nPrimary paper figures:"
    )

    print(
        "  01_micro_graph.png"
    )

    print(
        "  02_fiedler_vector.png"
    )

    print(
        "  03_effective_resistance.png"
    )

    print(
        "  03b_renormalised_effective_resistance.png"
    )

    print(
        "  03c_effective_resistance_comparison.png"
    )

    print(
        "  04_coarse_graph.png"
    )

    print(
        "  05_coarse_fiedler_vector.png"
    )

    print(
        "  06_rg_flow.png"
    )

    print(
        "\nAdditional analysis figures:"
    )

    print(
        "  07_eigenvalue_spectrum.png"
    )

    print(
        "  08_qfern_edge_sensitivity.png"
    )

    print(
        "  09_kuramoto_response.png"
    )

    print(
        "\nNumerical data:"
    )

    print(
        "  qfern_rg_data.npz"
    )

    print(
        "\nSummary:"
    )

    print(
        "  simulation_summary.txt"
    )

    print("=" * 75)
