import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_dataset(n_accounts=50, n_transactions=500, fraud_ratio=0.15, seed=42):
    np.random.seed(seed)

    accounts = [f"ACC_{i:04d}" for i in range(n_accounts)]
    locations = ["New York", "London", "Tokyo", "Singapore", "Dubai", "Zurich", "Hong Kong", "Sydney"]

    n_fraud_accounts = max(3, int(n_accounts * fraud_ratio))
    fraud_accounts = set(np.random.choice(accounts, n_fraud_accounts, replace=False))

    fraud_ring_1 = list(fraud_accounts)[:max(2, n_fraud_accounts // 2)]
    fraud_ring_2 = list(fraud_accounts)[max(2, n_fraud_accounts // 2):]

    transactions = []
    base_time = datetime(2024, 1, 1)

    for i in range(n_transactions):
        tx_id = f"TX_{i:06d}"
        timestamp = base_time + timedelta(hours=np.random.randint(0, 8760))

        if np.random.random() < 0.3 and len(fraud_ring_1) >= 2:
            sender = np.random.choice(fraud_ring_1)
            receiver = np.random.choice(fraud_ring_1)
            while receiver == sender:
                receiver = np.random.choice(fraud_ring_1)
            amount = np.random.lognormal(mean=8, sigma=1.5)
            fraud_label = 1
            location = np.random.choice(locations[:3])
            timestamp = base_time + timedelta(
                hours=np.random.randint(0, 720),
                minutes=np.random.randint(0, 60)
            )
        elif np.random.random() < 0.15 and len(fraud_ring_2) >= 2:
            sender = np.random.choice(fraud_ring_2)
            receiver = np.random.choice(fraud_ring_2)
            while receiver == sender:
                receiver = np.random.choice(fraud_ring_2)
            amount = np.random.lognormal(mean=7, sigma=2)
            fraud_label = 1
            location = np.random.choice(locations)
        else:
            sender = np.random.choice(accounts)
            receiver = np.random.choice(accounts)
            while receiver == sender:
                receiver = np.random.choice(accounts)
            amount = np.random.lognormal(mean=5, sigma=1)
            fraud_label = 1 if (sender in fraud_accounts or receiver in fraud_accounts) and np.random.random() < 0.3 else 0
            location = np.random.choice(locations)

        transactions.append({
            "transaction_id": tx_id,
            "sender_account": sender,
            "receiver_account": receiver,
            "amount": round(amount, 2),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "location": location,
            "fraud_label": fraud_label,
        })

    df = pd.DataFrame(transactions)
    return df
