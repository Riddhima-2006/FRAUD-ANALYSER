import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)


def run_isolation_forest(features: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    predictions = clf.fit_predict(features)
    scores = clf.decision_function(features)
    anomaly_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    logger.info(f"Isolation Forest: {(predictions == -1).sum()} anomalies detected")
    return anomaly_scores


def run_lof(features: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    clf = LocalOutlierFactor(n_neighbors=min(20, max(2, len(features) - 1)), contamination=contamination)
    predictions = clf.fit_predict(features)
    scores = clf.negative_outlier_factor_
    anomaly_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    logger.info(f"LOF: {(predictions == -1).sum()} anomalies detected")
    return anomaly_scores


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / degree
        support = torch.mm(adj_norm, x)
        out = self.linear(support)
        return out


class FraudGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super().__init__()
        self.conv1 = GraphConvLayer(input_dim, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.attention_weights = None

    def forward(self, x, adj):
        h = F.relu(self.conv1(x, adj))
        h = F.dropout(h, p=0.3, training=self.training)

        attn = torch.mm(h, h.t())
        attn = F.softmax(attn, dim=1)
        self.attention_weights = attn.detach()

        h = F.relu(self.conv2(h, adj))
        h = F.dropout(h, p=0.3, training=self.training)
        out = self.classifier(h)
        return out

    def get_node_importance(self, x, adj, node_idx):
        self.eval()
        with torch.no_grad():
            _ = self.forward(x, adj)
            if self.attention_weights is not None:
                importance = self.attention_weights[node_idx].numpy()
                return importance
        return None


def build_adjacency_matrix(G, node_list):
    n = len(node_list)
    node_idx = {node: i for i, node in enumerate(node_list)}
    adj = np.zeros((n, n))
    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            adj[node_idx[u]][node_idx[v]] = 1
            adj[node_idx[v]][node_idx[u]] = 1
    adj = adj + np.eye(n)
    return adj


def train_gnn(node_features: dict, G, labels: dict, epochs: int = 100):
    node_list = sorted(node_features.keys())
    n_nodes = len(node_list)
    if n_nodes < 3:
        logger.warning("Too few nodes for GNN training")
        return None, node_list, {}

    feature_dim = len(next(iter(node_features.values())))
    X = np.array([node_features[n] for n in node_list], dtype=np.float32)
    y = np.array([labels.get(n, 0) for n in node_list], dtype=np.int64)

    adj = build_adjacency_matrix(G, node_list)

    X_tensor = torch.FloatTensor(X)
    adj_tensor = torch.FloatTensor(adj)
    y_tensor = torch.LongTensor(y)

    n_fraud = (y == 1).sum()
    n_legit = (y == 0).sum()
    if n_fraud == 0:
        weight = torch.FloatTensor([1.0, 5.0])
    else:
        weight = torch.FloatTensor([1.0, max(1.0, n_legit / (n_fraud + 1))])

    model = FraudGNN(input_dim=feature_dim, hidden_dim=32, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=weight)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_tensor, adj_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor, adj_tensor)
        probs = F.softmax(logits, dim=1)
        gnn_scores = probs[:, 1].numpy()

    scores = {node_list[i]: float(gnn_scores[i]) for i in range(n_nodes)}
    logger.info(f"GNN trained for {epochs} epochs, final loss: {loss.item():.4f}")
    return model, node_list, scores


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> dict:
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
    logger.info(f"Model Metrics: {metrics}")
    return metrics
