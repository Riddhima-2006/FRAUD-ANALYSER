import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


def load_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    required = {"transaction_id", "sender_account", "receiver_account", "amount", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info(f"Loaded dataset with {len(df)} transactions and {len(df.columns)} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["timestamp"] = df["timestamp"].fillna(method="ffill")
    logger.info("Missing values handled")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    sender_freq = df.groupby("sender_account")["transaction_id"].transform("count")
    receiver_freq = df.groupby("receiver_account")["transaction_id"].transform("count")
    df["sender_tx_frequency"] = sender_freq
    df["receiver_tx_frequency"] = receiver_freq

    df["sender_avg_amount"] = df.groupby("sender_account")["amount"].transform("mean")
    df["receiver_avg_amount"] = df.groupby("receiver_account")["amount"].transform("mean")

    df["sender_std_amount"] = df.groupby("sender_account")["amount"].transform("std").fillna(0)

    df = df.sort_values(["sender_account", "timestamp"])
    df["time_gap"] = df.groupby("sender_account")["timestamp"].diff().dt.total_seconds().fillna(0)

    df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-8)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)

    if "location" in df.columns:
        le = LabelEncoder()
        df["location_encoded"] = le.fit_transform(df["location"].astype(str))
        df["sender_unique_locations"] = df.groupby("sender_account")["location"].transform("nunique")
    else:
        df["location_encoded"] = 0
        df["sender_unique_locations"] = 1

    logger.info("Feature engineering complete")
    return df


def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    scaler = MinMaxScaler()
    valid_cols = [c for c in feature_cols if c in df.columns]
    if valid_cols:
        df[valid_cols] = scaler.fit_transform(df[valid_cols].fillna(0))
    logger.info(f"Normalized {len(valid_cols)} features")
    return df


FEATURE_COLS = [
    "amount", "sender_tx_frequency", "receiver_tx_frequency",
    "sender_avg_amount", "receiver_avg_amount", "sender_std_amount",
    "time_gap", "amount_zscore", "hour_of_day", "day_of_week",
    "is_weekend", "is_night", "location_encoded", "sender_unique_locations"
]


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = load_and_validate(df)
    df = handle_missing_values(df)
    df = engineer_features(df)
    df = normalize_features(df, FEATURE_COLS)
    return df
