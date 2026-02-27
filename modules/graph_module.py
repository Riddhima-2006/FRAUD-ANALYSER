import networkx as nx
import numpy as np
import pandas as pd
import community as community_louvain
import logging

logger = logging.getLogger(__name__)


def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    accounts = set(df["sender_account"].unique()) | set(df["receiver_account"].unique())
    for acc in accounts:
        G.add_node(acc)

    for _, row in df.iterrows():
        sender = row["sender_account"]
        receiver = row["receiver_account"]
        amount = row["amount"]
        if G.has_edge(sender, receiver):
            G[sender][receiver]["weight"] += amount
            G[sender][receiver]["count"] += 1
        else:
            G.add_edge(sender, receiver, weight=amount, count=1)

    logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def compute_centrality_metrics(G: nx.DiGraph) -> dict:
    undirected = G.to_undirected()

    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(undirected, k=min(100, len(undirected.nodes)))
    closeness_cent = nx.closeness_centrality(undirected)
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

    try:
        clustering_coeff = nx.clustering(undirected)
    except Exception:
        clustering_coeff = {n: 0 for n in G.nodes()}

    metrics = {}
    for node in G.nodes():
        metrics[node] = {
            "degree_centrality": degree_cent.get(node, 0),
            "betweenness_centrality": betweenness_cent.get(node, 0),
            "closeness_centrality": closeness_cent.get(node, 0),
            "pagerank": pagerank.get(node, 0),
            "clustering_coefficient": clustering_coeff.get(node, 0),
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node),
        }

    logger.info(f"Computed centrality metrics for {len(metrics)} nodes")
    return metrics


def detect_communities(G: nx.DiGraph) -> dict:
    undirected = G.to_undirected()
    if len(undirected.nodes()) == 0:
        return {}
    partition = community_louvain.best_partition(undirected, random_state=42)
    logger.info(f"Detected {len(set(partition.values()))} communities")
    return partition


def get_community_details(partition: dict, centrality_metrics: dict, fraud_labels: dict = None) -> pd.DataFrame:
    records = []
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    for comm_id, members in communities.items():
        avg_pagerank = np.mean([centrality_metrics.get(m, {}).get("pagerank", 0) for m in members])
        avg_betweenness = np.mean([centrality_metrics.get(m, {}).get("betweenness_centrality", 0) for m in members])

        fraud_count = 0
        if fraud_labels:
            fraud_count = sum(1 for m in members if fraud_labels.get(m, 0) == 1)

        records.append({
            "community_id": comm_id,
            "size": len(members),
            "members": members,
            "avg_pagerank": avg_pagerank,
            "avg_betweenness": avg_betweenness,
            "fraud_count": fraud_count,
            "fraud_ratio": fraud_count / len(members) if fraud_labels else 0,
        })

    return pd.DataFrame(records).sort_values("fraud_ratio", ascending=False).reset_index(drop=True)


def get_node_features_from_graph(G: nx.DiGraph, centrality_metrics: dict, account_features: dict) -> dict:
    node_features = {}
    for node in G.nodes():
        cm = centrality_metrics.get(node, {})
        af = account_features.get(node, {})
        features = [
            cm.get("degree_centrality", 0),
            cm.get("betweenness_centrality", 0),
            cm.get("closeness_centrality", 0),
            cm.get("pagerank", 0),
            cm.get("clustering_coefficient", 0),
            cm.get("in_degree", 0),
            cm.get("out_degree", 0),
            af.get("avg_amount", 0),
            af.get("tx_frequency", 0),
            af.get("std_amount", 0),
            af.get("unique_locations", 0),
            af.get("avg_time_gap", 0),
        ]
        node_features[node] = features
    return node_features


def add_transaction_to_graph(G: nx.DiGraph, sender: str, receiver: str, amount: float) -> nx.DiGraph:
    if not G.has_node(sender):
        G.add_node(sender)
    if not G.has_node(receiver):
        G.add_node(receiver)

    if G.has_edge(sender, receiver):
        G[sender][receiver]["weight"] += amount
        G[sender][receiver]["count"] += 1
    else:
        G.add_edge(sender, receiver, weight=amount, count=1)

    return G
