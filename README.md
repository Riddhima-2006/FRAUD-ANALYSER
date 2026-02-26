# AI-Based Financial Fraud Intelligence System

## Overview
A full-stack fraud detection system that identifies fraud rings (not just individual transactions) using graph analytics, anomaly detection, and Graph Neural Networks (GNN). Built with Python and Streamlit.

## Recent Changes
- 2026-02-24: Initial build of complete fraud intelligence system with all modules

## Architecture

### Modular Structure
- `app.py` - Main Streamlit dashboard (UI layer)
- `modules/data_processing.py` - Data preprocessing, feature engineering, normalization
- `modules/graph_module.py` - NetworkX graph construction, centrality metrics, community detection
- `modules/ml_module.py` - Isolation Forest, LOF, GNN model training and evaluation
- `modules/risk_scoring.py` - Combined risk scoring and risk explanation
- `modules/sample_data.py` - Sample transaction data generator

### Key Technologies
- **Streamlit** - Interactive dashboard on port 5000
- **NetworkX** - Transaction graph construction and analysis
- **scikit-learn** - Isolation Forest, Local Outlier Factor
- **PyTorch** - Custom GNN implementation (GraphConvLayer)
- **Plotly** - Interactive visualizations
- **python-louvain** - Community detection (Louvain algorithm)

### Dashboard Tabs
1. Upload & Process - CSV upload or sample data, pipeline execution, model metrics
2. Transaction Graph - Interactive network visualization colored by risk
3. Fraud Clusters - Community detection results with fraud ratios
4. Top Risky Accounts - Ranked accounts with score breakdown and explanations
5. Anomaly Analysis - Score distributions (IF, LOF, GNN), scatter plots, pie charts
6. Live Scoring - Enter new transactions for real-time risk assessment

### Data Flow
CSV Upload -> Preprocessing -> Graph Building -> Centrality Metrics + Community Detection -> Anomaly Detection (IF + LOF) -> GNN Training -> Risk Score Computation -> Dashboard Display

### Risk Scoring
Combined score (0-1) from: Isolation Forest (20%), LOF (20%), Centrality (20%), GNN (40%)
Risk levels: Critical (>=0.7), High (>=0.5), Medium (>=0.3), Low (<0.3)

## Running
```bash
streamlit run app.py --server.port 5000
```
"# AI-Based-Financial-Fraud-Intelligence-System" 
