import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_risk_scores(
    accounts: list,
    anomaly_scores_if: dict,
    anomaly_scores_lof: dict,
    centrality_metrics: dict,
    gnn_scores: dict,
    weights: dict = None,
) -> pd.DataFrame:
    if weights is None:
        weights = {
            "anomaly_if": 0.2,
            "anomaly_lof": 0.2,
            "centrality": 0.2,
            "gnn": 0.4,
        }

    records = []
    for acc in accounts:
        if_score = anomaly_scores_if.get(acc, 0.0)
        lof_score = anomaly_scores_lof.get(acc, 0.0)

        cm = centrality_metrics.get(acc, {})
        centrality_score = (
            cm.get("pagerank", 0) * 0.3
            + cm.get("betweenness_centrality", 0) * 0.3
            + cm.get("degree_centrality", 0) * 0.2
            + cm.get("clustering_coefficient", 0) * 0.2
        )
        centrality_score = min(1.0, centrality_score * 5)

        gnn_score = gnn_scores.get(acc, 0.0)

        risk_score = (
            weights["anomaly_if"] * if_score
            + weights["anomaly_lof"] * lof_score
            + weights["centrality"] * centrality_score
            + weights["gnn"] * gnn_score
        )
        risk_score = float(np.clip(risk_score, 0, 1))

        records.append({
            "account": acc,
            "isolation_forest_score": round(if_score, 4),
            "lof_score": round(lof_score, 4),
            "centrality_score": round(centrality_score, 4),
            "gnn_score": round(gnn_score, 4),
            "risk_score": round(risk_score, 4),
            "risk_level": categorize_risk(risk_score),
        })

    df = pd.DataFrame(records).sort_values("risk_score", ascending=False).reset_index(drop=True)
    logger.info(f"Computed risk scores for {len(df)} accounts")
    return df


def categorize_risk(score: float) -> str:
    if score >= 0.7:
        return "Critical"
    elif score >= 0.5:
        return "High"
    elif score >= 0.3:
        return "Medium"
    else:
        return "Low"


def explain_risk(account: str, risk_df: pd.DataFrame, centrality_metrics: dict, community_df: pd.DataFrame = None) -> dict:
    row = risk_df[risk_df["account"] == account]
    if row.empty:
        return {"account": account, "explanation": "No data available"}

    row = row.iloc[0]
    factors = []

    components = {
        "Isolation Forest Anomaly": row["isolation_forest_score"],
        "Local Outlier Factor": row["lof_score"],
        "Graph Centrality": row["centrality_score"],
        "GNN Prediction": row["gnn_score"],
    }

    sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)

    for name, score in sorted_components:
        if score > 0.5:
            factors.append(f"{name}: {score:.2f} (high)")
        elif score > 0.3:
            factors.append(f"{name}: {score:.2f} (moderate)")

    cm = centrality_metrics.get(account, {})
    if cm.get("betweenness_centrality", 0) > 0.1:
        factors.append("Acts as a bridge between different account groups")
    if cm.get("pagerank", 0) > 0.05:
        factors.append("High influence in the transaction network")
    if cm.get("in_degree", 0) > 5:
        factors.append(f"Receives transactions from {cm['in_degree']} different accounts")

    return {
        "account": account,
        "risk_score": row["risk_score"],
        "risk_level": row["risk_level"],
        "top_factors": sorted_components[:3],
        "explanation": factors,
    }
