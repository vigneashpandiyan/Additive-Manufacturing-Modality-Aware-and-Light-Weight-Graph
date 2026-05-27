# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Constructing fully connected and optimized spring graph layout representations to map multi-modal interactions.

Note: Any reuse of this code should be authorized by the code author.
"""

import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib.lines import Line2D


def plot_temporal_graph_representation():
    """
    Description:
        Constructs and visualizes a linear temporal graph representation with sequential nodes and directed edges, marking sensory modalities inside nodes, and saves the plot as a PNG.
    Purpose:
        To save illustrative linear graph representation figures.
    Input Types:
        - None
    Output Types:
        - None
    """
    total_nodes = 8
    G = nx.DiGraph()
    G.add_nodes_from(range(1, 9))
    G.add_edges_from([(i, i + 1) for i in range(1, 8)])

    # Position nodes in a straight line for temporal progression
    pos = {i: (i - 1, 0) for i in G.nodes()}

    # Node color palette
    node_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#17becf']

    plt.figure(figsize=(12, 3), dpi=300)

    # Draw nodes with colored borders
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        edgecolors='black',
        linewidths=2,
        node_size=1800
    )

    # Draw directed edges between nodes
    nx.draw_networkx_edges(
        G, pos,
        edge_color='black',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=20,
        width=2.5,
        connectionstyle='arc3,rad=0.0'
    )

    # Add channel indicators (▲ for acoustic, ▼ for back-reflection)
    for i in G.nodes():
        x, y = pos[i]
        plt.text(
            x, y + 0.0005, '▲',
            ha='center', va='center',
            fontsize=14, color='#FF5722'
        )
        plt.text(
            x, y - 0.0005, '▼',
            ha='center', va='center',
            fontsize=14, color='#3F51B5'
        )

    # Add node identifiers
    for i in G.nodes():
        x, y = pos[i]
        plt.text(
            x, y - 0.004, f"Node {i}",
            ha='center', fontsize=16
        )

    # Add time window labels
    label_offset = 0.002
    for i in range(1, total_nodes + 1):
        x, y = pos[i]
        plt.text(
            x, y + label_offset,
            f"t={i-1}▲t \n to \n t={i}▲t",
            ha='center',
            va='bottom',
            fontsize=16,
            color='black',
            alpha=1
        )

    # Add temporal progression annotation
    plt.annotate(
        'Temporal Progression →',
        xy=(0.5, -0.0000005),
        xycoords='axes fraction',
        ha='center',
        va='center',
        fontsize=16,
        rotation=0
    )

    plt.title("Temporal Graph Representation", fontsize=20, pad=30)
    plt.axis('off')
    plt.tight_layout()

    # Create output directory if needed
    os.makedirs("Figures", exist_ok=True)
    plt.savefig("Figures/temporal_graph.png", dpi=300, bbox_inches='tight')
    plt.show()


def create_fully_connected_graph():
    """
    Description:
        Constructs and visualizes a fully connected temporal graph with bidirectional edges, highlighting forward (green) and backward (purple) temporal dependencies.
    Purpose:
        To save illustrative representations of fully connected spatial structures.
    Input Types:
        - None
    Output Types:
        - None
    """
    # === Parameters ===
    num_primary_nodes = 8
    total_nodes = num_primary_nodes

    # Generate all possible edges (excluding self-loops)
    all_edges = []
    for i in range(1, total_nodes + 1):
        for j in range(1, total_nodes + 1):
            if i != j:
                all_edges.append((i, j))  # i → j

    # Color edges based on direction
    edge_colors = []
    for u, v in all_edges:
        if u < v:
            edge_colors.append('#4CAF50')  # Forward connection (green)
        else:
            edge_colors.append('#9C27B0')  # Backward connection (violet)

    # === Create Graph ===
    G = nx.DiGraph()
    for i in range(1, total_nodes + 1):
        G.add_node(i)
    G.add_edges_from(all_edges)

    # Position nodes diagonally for temporal representation
    pos = {}
    x_spacing = 1.0 / (total_nodes - 1)
    y_spacing = 1.0 / (total_nodes - 1)
    for i in range(1, total_nodes + 1):
        pos[i] = ((i-1) * x_spacing, (i-1) * y_spacing)

    # === Plot ===
    plt.figure(figsize=(12, 8), dpi=100)

    # Node colors
    node_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                   '#98D8C8', '#B39DDB', '#AED581', '#FFD54F']

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        edgecolors='black',
        linewidths=2,
        node_size=3500,
        alpha=0.9
    )

    # Draw edges with direction-specific colors
    for (u, v), color in zip(all_edges, edge_colors):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=color,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
            width=2.0,
            connectionstyle='arc3,rad=0.2',
            alpha=0.7
        )

    # Add channel indicators as triangles
    for i in range(1, total_nodes + 1):
        x, y = pos[i]
        plt.text(
            x, y + 0.03, '▲',
            ha='center', va='center',
            fontsize=14, color='#FF5722'
        )
        plt.text(
            x, y - 0.03, '▼',
            ha='center', va='center',
            fontsize=14, color='#3F51B5'
        )

    # Add node labels with time information
    label_offset = 0.15
    for i in range(1, total_nodes + 1):
        x, y = pos[i]
        plt.text(
            x, y + label_offset,
            f"Node {i}\nt={(i-1)}Δt to t={i}Δt",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                boxstyle='round,pad=0.4',
                alpha=0.9
            )
        )

    # Create legend elements
    legend_elements = [
        Line2D([0], [0], color='#4CAF50', lw=3, label='Forward (i→j)'),
        Line2D([0], [0], color='#9C27B0', lw=3, label='Backward (j→i)'),
        Line2D([0], [0], marker='^', color='w', label='Acoustic (Ch1)',
               markerfacecolor='#FF5722', markersize=15),
        Line2D([0], [0], marker='v', color='w', label='Back reflection (Ch2)',
               markerfacecolor='#3F51B5', markersize=15)
    ]

    # Add legend
    plt.legend(
        handles=legend_elements,
        loc='upper left',
        framealpha=0.9,
        prop={'size': 12}
    )

    # Add temporal progression annotation
    plt.annotate(
        'Temporal Progression →',
        xy=(0.5, 0.35),
        xycoords='axes fraction',
        ha='center',
        va='center',
        fontsize=14,
        rotation=30
    )

    # Adjust plot boundaries
    plt.xlim(-0.1, 1.2)
    plt.ylim(-0.1, 1.2)
    plt.axis('off')
    plt.tight_layout()

    # Save and show
    os.makedirs("Figures", exist_ok=True)
    plt.savefig("Figures/fully_connected_graph.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def create_optimized_spring_graph():
    """
    Description:
        Constructs and visualizes a fully connected graph utilizing spring layout optimizations to map window relationships with custom arrow offsets.
    Purpose:
        To save illustrative optimized spring layout representations.
    Input Types:
        - None
    Output Types:
        - None
    """
    def draw_outside_only_arrows(G, pos, ax, node_size=3500):
        """
        Description:
            Draws arrows between nodes without overlapping node boundaries by offsetting coordinates according to node size.
        Purpose:
            To prevent visual arrow-node overlaps.
        Input Types:
            - G (networkx.DiGraph): Target directed graph.
            - pos (dict): Positions map dictionary.
            - ax (matplotlib.axes.Axes): Target axis plot object.
            - node_size (int): Size of nodes. Default is 3500.
        Output Types:
            - None: Draws patches onto the axis directly.
        """
        radius = np.sqrt(node_size) / 2000
        safety_margin = 0.1

        for (u, v), color in zip(G.edges(), edge_colors):
            if u == v:
                continue  # Skip self-loops

            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx, dy = x2 - x1, y2 - y1
            distance = np.sqrt(dx**2 + dy**2)

            if distance == 0 or distance <= 2 * radius:
                continue  # Skip nodes too close

            dx /= distance
            dy /= distance

            # Calculate arrow start and end positions
            start_x = x1 + dx * (radius + safety_margin)
            start_y = y1 + dy * (radius + safety_margin)
            end_x = x2 - dx * (radius + safety_margin)
            end_y = y2 - dy * (radius + safety_margin)

            # Create arrow patch
            arrow = mpatches.FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle='-|>',
                connectionstyle="arc3,rad=0.2",
                mutation_scale=20,
                color=color,
                lw=3,
                alpha=0.95
            )
            ax.add_patch(arrow)

    # === Graph Setup ===
    G = nx.DiGraph()
    num_nodes = 8
    G.add_nodes_from(range(1, num_nodes + 1))
    edges = [(i, j) for i in range(1, num_nodes + 1)
             for j in range(1, num_nodes + 1) if i != j]
    G.add_edges_from(edges)

    # Node positions (spring layout)
    pos = nx.spring_layout(G, seed=42)

    # Node colors
    node_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#17becf']

    # Map edges to source node colors
    source_node_color_map = {i: color for i,
                             color in zip(G.nodes(), node_colors)}
    edge_colors = [source_node_color_map[u] for u, v in G.edges()]

    # Create legend elements
    legend_elements = [
        Line2D([0], [0], color=color, lw=3, label=f'From Node {i}')
        for i, color in zip(G.nodes(), node_colors)
    ] + [
        Line2D([0], [0], marker='^', color='w', label='Acoustic (Ch1)',
               markerfacecolor='#FF5722', markersize=15),
        Line2D([0], [0], marker='v', color='w', label='Back reflection (Ch2)',
               markerfacecolor='#3F51B5', markersize=15)
    ]

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(18, 9), dpi=300)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        edgecolors='black',
        linewidths=2.5,
        node_size=3500,
        alpha=0.9,
        ax=ax
    )

    # Draw optimized arrows
    draw_outside_only_arrows(G, pos, ax)

    # Add channel indicators
    for i in G.nodes():
        x, y = pos[i]
        ax.text(
            x, y + 0.03, '▲',
            ha='center', va='center',
            fontsize=14, color='#FF5722'
        )
        ax.text(
            x, y - 0.03, '▼',
            ha='center', va='center',
            fontsize=14, color='#3F51B5'
        )
        ax.text(
            x, y + 0.16,
            f"Node {i}\nt={(i-1)}▲t to t={i}▲t",
            ha='center', va='bottom',
            fontsize=24,
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                boxstyle='round,pad=0.4',
                alpha=0.2
            )
        )

    # Add legend
    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.18, 0.4),
        framealpha=0.2,
        prop={'size': 20}
    )

    # Final styling
    ax.axis('off')
    plt.tight_layout()

    # Save and show
    os.makedirs("Figures", exist_ok=True)
    plt.savefig("Figures/spring_layout_graph.png",
                dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Execute all visualization functions
    create_fully_connected_graph()
    create_optimized_spring_graph()
    plot_temporal_graph_representation()
