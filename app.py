import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import io
import logging
import warnings

warnings.filterwarnings("ignore")

from modules.data_processing import preprocess_pipeline, FEATURE_COLS
from modules.graph_module import (
    build_transaction_graph,
    compute_centrality_metrics,
    detect_communities,
    get_community_details,
    get_node_features_from_graph,
    add_transaction_to_graph,
)
from modules.ml_module import (
    run_isolation_forest,
    run_lof,
    train_gnn,
    evaluate_model,
)
from modules.risk_scoring import compute_risk_scores, explain_risk
from modules.sample_data import generate_sample_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AIBANK Analyst", layout="wide")


def get_account_features(df):
    features = {}
    for acc in set(df["sender_account"].unique()) | set(df["receiver_account"].unique()):
        sender_txs = df[df["sender_account"] == acc]
        receiver_txs = df[df["receiver_account"] == acc]
        all_amounts = pd.concat([sender_txs["amount"], receiver_txs["amount"]])
        features[acc] = {
            "avg_amount": all_amounts.mean() if len(all_amounts) > 0 else 0,
            "tx_frequency": len(sender_txs) + len(receiver_txs),
            "std_amount": all_amounts.std() if len(all_amounts) > 1 else 0,
            "unique_locations": sender_txs["sender_unique_locations"].max() if "sender_unique_locations" in sender_txs.columns and len(sender_txs) > 0 else 1,
            "avg_time_gap": sender_txs["time_gap"].mean() if "time_gap" in sender_txs.columns and len(sender_txs) > 0 else 0,
        }
    return features


def get_account_anomaly_features(df, accounts):
    records = []
    for acc in accounts:
        sender_txs = df[df["sender_account"] == acc]
        receiver_txs = df[df["receiver_account"] == acc]
        all_txs = pd.concat([sender_txs, receiver_txs])

        feat = [
            all_txs["amount"].mean() if len(all_txs) > 0 else 0,
            all_txs["amount"].std() if len(all_txs) > 1 else 0,
            len(sender_txs),
            len(receiver_txs),
            all_txs["time_gap"].mean() if "time_gap" in all_txs.columns and len(all_txs) > 0 else 0,
            all_txs["amount_zscore"].mean() if "amount_zscore" in all_txs.columns and len(all_txs) > 0 else 0,
            all_txs["is_night"].mean() if "is_night" in all_txs.columns and len(all_txs) > 0 else 0,
            all_txs["is_weekend"].mean() if "is_weekend" in all_txs.columns and len(all_txs) > 0 else 0,
        ]
        records.append(feat)
    return np.array(records, dtype=np.float32)


def run_full_pipeline(df):
    with st.spinner("Preprocessing data..."):
        df = preprocess_pipeline(df)

    with st.spinner("Building transaction graph..."):
        G = build_transaction_graph(df)

    with st.spinner("Computing graph metrics..."):
        centrality_metrics = compute_centrality_metrics(G)
        communities = detect_communities(G)

    accounts = sorted(G.nodes())
    account_features = get_account_features(df)

    with st.spinner("Running anomaly detection..."):
        anomaly_features = get_account_anomaly_features(df, accounts)
        np.nan_to_num(anomaly_features, copy=False)
        if_scores = run_isolation_forest(anomaly_features)
        lof_scores = run_lof(anomaly_features)
        if_dict = {accounts[i]: float(if_scores[i]) for i in range(len(accounts))}
        lof_dict = {accounts[i]: float(lof_scores[i]) for i in range(len(accounts))}

    fraud_labels = {}
    has_labels = "fraud_label" in df.columns
    if has_labels:
        for _, row in df.iterrows():
            if row["fraud_label"] == 1:
                fraud_labels[row["sender_account"]] = 1
                fraud_labels[row["receiver_account"]] = 1
        for acc in accounts:
            fraud_labels.setdefault(acc, 0)

    with st.spinner("Training GNN model..."):
        node_features = get_node_features_from_graph(G, centrality_metrics, account_features)
        model, node_list, gnn_scores = train_gnn(node_features, G, fraud_labels, epochs=150)

    with st.spinner("Computing risk scores..."):
        risk_df = compute_risk_scores(accounts, if_dict, lof_dict, centrality_metrics, gnn_scores)

    community_df = get_community_details(communities, centrality_metrics, fraud_labels if has_labels else None)

    metrics = {}
    if has_labels and model is not None:
        y_true = np.array([fraud_labels.get(acc, 0) for acc in accounts])
        y_scores = np.array([gnn_scores.get(acc, 0) for acc in accounts])
        y_pred = (y_scores > 0.5).astype(int)
        metrics = evaluate_model(y_true, y_pred, y_scores)

    return {
        "df": df,
        "G": G,
        "centrality_metrics": centrality_metrics,
        "communities": communities,
        "community_df": community_df,
        "risk_df": risk_df,
        "if_dict": if_dict,
        "lof_dict": lof_dict,
        "gnn_scores": gnn_scores,
        "fraud_labels": fraud_labels,
        "model": model,
        "node_features": node_features,
        "node_list": node_list if model else accounts,
        "metrics": metrics,
        "account_features": account_features,
    }


def plot_transaction_graph(G, communities, risk_df, max_nodes=200):
    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        pagerank = nx.pagerank(G)
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        nodes = [n[0] for n in top_nodes]
        G = G.subgraph(nodes).copy()

    pos = nx.spring_layout(G, seed=42, k=2 / np.sqrt(max(len(G.nodes()), 1)))

    risk_map = dict(zip(risk_df["account"], risk_df["risk_score"]))
    risk_level_map = dict(zip(risk_df["account"], risk_df["risk_level"]))

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
    )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_color = [risk_map.get(n, 0) for n in G.nodes()]
    node_text = [
        f"Account: {n}<br>Risk: {risk_map.get(n, 0):.3f}<br>Level: {risk_level_map.get(n, 'N/A')}<br>Community: {communities.get(n, 'N/A')}"
        for n in G.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers",
        hoverinfo="text", text=node_text,
        marker=dict(
            showscale=True, colorscale="RdYlGn_r",
            color=node_color, size=10,
            colorbar=dict(thickness=15, title="Risk Score", xanchor="left"),
            line=dict(width=1, color="#333"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            title="Transaction Network Graph (colored by risk score)",
        ),
    )
    return fig


def main():
    st.title(" Smart Fraud Intelligence Dashboard")
    st.subheader("Real-Time AI Risk Monitoring System")
    st.markdown("Detect fraud rings and suspicious accounts using graph analytics, anomaly detection, and GNN models.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Upload & Process", "Transaction Graph", "Fraud Clusters",
        "Top Risky Accounts", "Anomaly Analysis", "Live Scoring"
    ])

    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None

    with tab1:
        st.header("Upload Transaction Dataset")
        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader("Upload CSV file with transaction data", type=["csv"])

        with col2:
            st.markdown("**Required columns:**")
            st.code("transaction_id, sender_account,\nreceiver_account, amount, timestamp")
            st.markdown("**Optional:** `location`, `fraud_label`")

        use_sample = st.button("Use Sample Dataset")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} transactions with {len(df.columns)} columns")
                st.dataframe(df.head(10), use_container_width=True)
                if st.button("Run Analysis Pipeline"):
                    results = run_full_pipeline(df)
                    st.session_state.pipeline_results = results
                    st.success("Analysis complete! Navigate to other tabs to explore results.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

        elif use_sample:
            with st.spinner("Generating sample dataset..."):
                df = generate_sample_dataset(n_accounts=60, n_transactions=600)
                st.success(f"Generated {len(df)} sample transactions across {df['sender_account'].nunique() + df['receiver_account'].nunique()} accounts")
                st.dataframe(df.head(10), use_container_width=True)
                results = run_full_pipeline(df)
                st.session_state.pipeline_results = results
                st.success("Analysis complete! Navigate to other tabs to explore results.")
                st.rerun()

        if st.session_state.pipeline_results:
            res = st.session_state.pipeline_results
            st.divider()
            st.subheader("Pipeline Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Transactions", len(res["df"]))
            c2.metric("Total Accounts", len(res["G"].nodes()))
            c3.metric("Graph Edges", len(res["G"].edges()))
            c4.metric("Communities Detected", len(res["community_df"]))

            if res["metrics"]:
                st.subheader("Model Evaluation Metrics")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Precision", f"{res['metrics'].get('precision', 0):.3f}")
                mc2.metric("Recall", f"{res['metrics'].get('recall', 0):.3f}")
                mc3.metric("F1-Score", f"{res['metrics'].get('f1_score', 0):.3f}")
                mc4.metric("ROC-AUC", f"{res['metrics'].get('roc_auc', 0):.3f}")

    with tab2:
        st.header("Transaction Network Graph")
        if st.session_state.pipeline_results is None:
            st.info("Please upload and process a dataset first.")
        else:
            res = st.session_state.pipeline_results
            fig = plot_transaction_graph(res["G"], res["communities"], res["risk_df"])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Graph Statistics")
            gc1, gc2, gc3 = st.columns(3)
            gc1.metric("Nodes", res["G"].number_of_nodes())
            gc2.metric("Edges", res["G"].number_of_edges())
            density = nx.density(res["G"])
            gc3.metric("Density", f"{density:.4f}")

    with tab3:
        st.header("Detected Fraud Clusters")
        if st.session_state.pipeline_results is None:
            st.info("Please upload and process a dataset first.")
        else:
            res = st.session_state.pipeline_results
            comm_df = res["community_df"]

            suspicious = comm_df[
                (comm_df["fraud_ratio"] > 0.3) | (comm_df["avg_pagerank"] > comm_df["avg_pagerank"].quantile(0.75))
            ].copy()

            if len(suspicious) > 0:
                st.warning(f"Found {len(suspicious)} suspicious clusters (potential fraud rings)")
            else:
                st.info("No strongly suspicious clusters detected. Showing all communities.")
                suspicious = comm_df

            for _, row in suspicious.iterrows():
                risk_color = "red" if row["fraud_ratio"] > 0.5 else "orange" if row["fraud_ratio"] > 0.2 else "green"
                with st.expander(f"Community {row['community_id']} — {row['size']} accounts | Fraud Ratio: {row['fraud_ratio']:.1%}", expanded=row["fraud_ratio"] > 0.3):
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Members", row["size"])
                    cc2.metric("Fraud Ratio", f"{row['fraud_ratio']:.1%}")
                    cc3.metric("Avg PageRank", f"{row['avg_pagerank']:.4f}")

                    members = row["members"]
                    member_risks = res["risk_df"][res["risk_df"]["account"].isin(members)].sort_values("risk_score", ascending=False)
                    st.dataframe(member_risks[["account", "risk_score", "risk_level", "gnn_score", "isolation_forest_score"]], use_container_width=True)

            st.divider()
            st.subheader("All Communities Overview")
            display_df = comm_df.drop(columns=["members"]).copy()
            fig_comm = px.bar(
                display_df, x="community_id", y="size",
                color="fraud_ratio", color_continuous_scale="RdYlGn_r",
                title="Community Sizes (colored by fraud ratio)",
                labels={"community_id": "Community", "size": "Number of Accounts"},
            )
            st.plotly_chart(fig_comm, use_container_width=True)

    with tab4:
        st.header("Top Risky Accounts")
        if st.session_state.pipeline_results is None:
            st.info("Please upload and process a dataset first.")
        else:
            res = st.session_state.pipeline_results
            risk_df = res["risk_df"]

            top_n = st.slider("Number of top risky accounts", 5, 50, 20)
            top_risky = risk_df.head(top_n)

            fig_risk = px.bar(
                top_risky, x="account", y="risk_score",
                color="risk_level",
                color_discrete_map={"Critical": "#d32f2f", "High": "#f57c00", "Medium": "#fbc02d", "Low": "#388e3c"},
                title=f"Top {top_n} Risky Accounts",
            )
            fig_risk.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_risk, use_container_width=True)

            st.subheader("Risk Score Breakdown")
            st.dataframe(
                top_risky[["account", "risk_score", "risk_level", "isolation_forest_score", "lof_score", "centrality_score", "gnn_score"]],
                use_container_width=True,
            )

            st.subheader("Account Risk Explanation")
            selected_account = st.selectbox("Select account for detailed explanation", top_risky["account"].tolist())
            if selected_account:
                explanation = explain_risk(selected_account, risk_df, res["centrality_metrics"], res["community_df"])
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.metric("Risk Score", f"{explanation['risk_score']:.4f}")
                    st.metric("Risk Level", explanation["risk_level"])
                with ec2:
                    st.markdown("**Top Contributing Factors:**")
                    for name, score in explanation.get("top_factors", []):
                        bar_pct = int(score * 100)
                        st.markdown(f"- **{name}**: {score:.3f}")
                        st.progress(min(score, 1.0))

                if explanation.get("explanation"):
                    st.markdown("**Why this account is flagged:**")
                    for exp in explanation["explanation"]:
                        st.markdown(f"- {exp}")

    with tab5:
        st.header("Anomaly Distribution Analysis")
        if st.session_state.pipeline_results is None:
            st.info("Please upload and process a dataset first.")
        else:
            res = st.session_state.pipeline_results
            risk_df = res["risk_df"]

            ac1, ac2 = st.columns(2)
            with ac1:
                fig_if = px.histogram(
                    risk_df, x="isolation_forest_score", nbins=30,
                    title="Isolation Forest Score Distribution",
                    color_discrete_sequence=["#1f77b4"],
                )
                st.plotly_chart(fig_if, use_container_width=True)
            with ac2:
                fig_lof = px.histogram(
                    risk_df, x="lof_score", nbins=30,
                    title="Local Outlier Factor Score Distribution",
                    color_discrete_sequence=["#ff7f0e"],
                )
                st.plotly_chart(fig_lof, use_container_width=True)

            fig_scatter = px.scatter(
                risk_df, x="isolation_forest_score", y="lof_score",
                color="risk_level", size="risk_score",
                color_discrete_map={"Critical": "#d32f2f", "High": "#f57c00", "Medium": "#fbc02d", "Low": "#388e3c"},
                title="Anomaly Score Comparison (IF vs LOF)",
                hover_data=["account", "risk_score"],
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            fig_gnn = px.histogram(
                risk_df, x="gnn_score", nbins=30,
                title="GNN Fraud Probability Distribution",
                color_discrete_sequence=["#9467bd"],
            )
            st.plotly_chart(fig_gnn, use_container_width=True)

            st.subheader("Risk Level Distribution")
            risk_counts = risk_df["risk_level"].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values, names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={"Critical": "#d32f2f", "High": "#f57c00", "Medium": "#fbc02d", "Low": "#388e3c"},
                title="Account Risk Distribution",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab6:
        st.header("Live Fraud Risk Scoring")
        if st.session_state.pipeline_results is None:
            st.info("Please upload and process a dataset first.")
        else:
            res = st.session_state.pipeline_results
            st.markdown("Enter a new transaction to dynamically update the graph and get real-time risk prediction.")

            with st.form("new_transaction"):
                fc1, fc2 = st.columns(2)
                with fc1:
                    new_sender = st.text_input("Sender Account", placeholder="e.g., ACC_0001 or NEW_SENDER")
                    new_amount = st.number_input("Amount", min_value=0.01, value=1000.0, step=100.0)
                    new_location = st.text_input("Location", value="New York")
                with fc2:
                    new_receiver = st.text_input("Receiver Account", placeholder="e.g., ACC_0010 or NEW_RECEIVER")
                    new_timestamp = st.text_input("Timestamp", value="2024-06-15 14:30:00")

                submitted = st.form_submit_button("Score Transaction")

            if submitted and new_sender and new_receiver:
                with st.spinner("Updating graph and scoring..."):
                    new_tx = pd.DataFrame([{
                        "transaction_id": f"TX_LIVE_{np.random.randint(100000, 999999)}",
                        "sender_account": new_sender,
                        "receiver_account": new_receiver,
                        "amount": new_amount,
                        "timestamp": new_timestamp,
                        "location": new_location,
                        "fraud_label": 0,
                    }])
                    updated_df = pd.concat([res["df"], preprocess_pipeline(new_tx)], ignore_index=True)

                    G = build_transaction_graph(updated_df)

                    new_centrality = compute_centrality_metrics(G)
                    new_communities = detect_communities(G)

                    accounts = sorted(G.nodes())
                    account_features = get_account_features(updated_df)

                    anomaly_feats = get_account_anomaly_features(updated_df, accounts)
                    np.nan_to_num(anomaly_feats, copy=False)

                    new_if = run_isolation_forest(anomaly_feats)
                    new_lof = run_lof(anomaly_feats)
                    if_dict = {accounts[i]: float(new_if[i]) for i in range(len(accounts))}
                    lof_dict = {accounts[i]: float(new_lof[i]) for i in range(len(accounts))}

                    node_feats = get_node_features_from_graph(G, new_centrality, account_features)
                    _, _, new_gnn_scores = train_gnn(node_feats, G, res["fraud_labels"], epochs=50)

                    new_risk_df = compute_risk_scores(accounts, if_dict, lof_dict, new_centrality, new_gnn_scores)

                from modules.risk_scoring import explain_risk as _explain

                st.subheader("Transaction Risk Assessment")

                for acc in [new_sender, new_receiver]:
                    acc_row = new_risk_df[new_risk_df["account"] == acc]
                    if not acc_row.empty:
                        score = acc_row.iloc[0]["risk_score"]
                        level = acc_row.iloc[0]["risk_level"]
                        color = {"Critical": "red", "High": "orange", "Medium": "yellow", "Low": "green"}.get(level, "gray")

                        st.markdown(f"### Account: `{acc}`")
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("Risk Score", f"{score:.4f}")
                        rc2.metric("Risk Level", level)
                        rc3.metric("IF Score", f"{acc_row.iloc[0]['isolation_forest_score']:.4f}")
                        rc4.metric("GNN Score", f"{acc_row.iloc[0]['gnn_score']:.4f}")

                        explanation = _explain(acc, new_risk_df, new_centrality)
                        if explanation.get("explanation"):
                            with st.expander("Why this account is flagged"):
                                for exp in explanation["explanation"]:
                                    st.markdown(f"- {exp}")
                        st.divider()


if __name__ == "__main__":
    main()
