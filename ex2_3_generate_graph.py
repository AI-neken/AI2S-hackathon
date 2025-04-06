#!/usr/bin/env python

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# Configuration
INPUT_PATH_2 = Path('outputs/02_output_shipments_1239.csv')
INPUT_PATH_3 = Path('outputs/03_output_shipments_1239.csv')
OUTPUT_DIR = Path('outputs')

def group_track(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shipment quantity by origin and destination."""
    return df.groupby(["Origin", "Destination"]).agg({"Quantity": "sum"}).reset_index()

def create_graph(df: pd.DataFrame, name: str = "graph") -> None:
    """Create and save a directed graph of trade flows based on quantity."""
    G = nx.DiGraph()

    # Build graph edges with weights
    for _, row in df.iterrows():
        origin = row['Origin']
        destination = row['Destination']
        quantity = row['Quantity']
        G.add_edge(origin, destination, weight=quantity)

    # Define edge thickness based on quantity
    weights = [G[u][v]['weight'] / 1000000 for u, v in G.edges()]  # Adjust divisor to scale

    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, width=weights, arrowstyle='->', arrowsize=20)

    plt.title('Trade Flow Graph Based on Quantity')
    plt.axis('off')
    plt.tight_layout()

    # Save graph as image
    output_path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved to {output_path}")

def main() -> None:
    """Main execution function for reading data and generating graphs."""
    # Load and group data
    df_ex_2 = pd.read_csv(INPUT_PATH_2)
    df_ex_3 = pd.read_csv(INPUT_PATH_3)

    df_ex_2_grouped = group_track(df_ex_2)
    df_ex_3_grouped = group_track(df_ex_3)

    # Create graphs
    create_graph(df_ex_2_grouped, name="02_output_shipments_1239")
    create_graph(df_ex_3_grouped, name="03_output_shipments_1239")

if __name__ == "__main__":
    main()