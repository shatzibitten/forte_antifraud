import pandas as pd
import numpy as np
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineerAgent:
    def __init__(self, df):
        self.df = df.copy()

    def generate_features(self):
        """Generates all features."""
        logger.info("Generating features...")
        self.generate_velocity_features()
        self.generate_interaction_features()
        self.generate_identity_features()
        self.generate_graph_features()
        self.generate_time_delta_features()
        self.generate_gnn_features()
        self.generate_anomaly_features()
        return self.df

    def generate_velocity_features(self):
        """Generates velocity features (rolling window stats)."""
        logger.info("Generating velocity features...")
        
        # Ensure sorted by time
        self.df = self.df.sort_values('transdatetime')
        
        # Group by client
        grouped = self.df.groupby('cst_dim_id')
        
        # Rolling windows need index to be datetime for time-based rolling
        # But we have multiple clients, so we iterate or use transform
        # Using transform with rolling is tricky with time offsets if index is not unique
        # Easier approach: set index to transdatetime, group by client, then rolling
        
        temp_df = self.df.set_index('transdatetime').sort_index()
        grouped_time = temp_df.groupby('cst_dim_id')
        
        # 1 hour count
        # We use '1h' string for time offset
        self.df['count_txn_1h'] = grouped_time['amount'].rolling('1h').count().reset_index(level=0, drop=True).values
        
        # 24 hour sum
        self.df['sum_amount_24h'] = grouped_time['amount'].rolling('24h').sum().reset_index(level=0, drop=True).values
        
        # 7 days std
        self.df['std_amount_7d'] = grouped_time['amount'].rolling('7d').std().reset_index(level=0, drop=True).values
        
        # Ratio of current amount to 30-day average (simulated by 30d rolling mean)
        self.df['avg_amount_30d'] = grouped_time['amount'].rolling('30d').mean().reset_index(level=0, drop=True).values
        self.df['amount_to_avg_30d'] = self.df['amount'] / (self.df['avg_amount_30d'] + 1e-5)

    def generate_interaction_features(self):
        """Generates interaction features between transaction and behavior."""
        logger.info("Generating interaction features...")
        
        # Amount / Avg Login Interval
        # avg_login_interval_30d is in seconds (presumably)
        self.df['amount_per_login_interval'] = self.df['amount'] / (self.df['avg_login_interval_30d'] + 1e-5)

        # --- NEW INTERACTIONS (v2) ---
        
        # 1. Night Time Flag (00:00 - 06:00)
        # Extract hour from transdatetime
        self.df['hour'] = self.df['transdatetime'].dt.hour
        self.df['is_night'] = ((self.df['hour'] >= 0) & (self.df['hour'] < 6)).astype(int)
        
        # 2. Night * Amount (High amount at night is suspicious)
        self.df['interaction_night_amount'] = self.df['is_night'] * self.df['amount']
        
        # 3. Device Changes * Frequency (New device + high velocity = suspicious)
        # monthly_os_changes * count_txn_1h
        self.df['interaction_device_freq'] = self.df['monthly_os_changes'] * self.df['count_txn_1h']
        
        # 4. Amount relative to history * Night (Spike in amount during night)
        self.df['interaction_night_rel_amount'] = self.df['is_night'] * self.df['amount_to_avg_30d']

    def generate_identity_features(self):
        """Generates identity consistency features."""
        logger.info("Generating identity features...")
        
        # High risk device: monthly_os_changes > 1
        self.df['high_risk_device'] = (self.df['monthly_os_changes'] > 1).astype(int)
        
        # Fake OS version (Hard Rule from TOR)
        # "iOS 26.0" etc.
        # We can check for "iOS/2" or similar impossible versions
        # For now, let's just flag if it contains "iOS/2" or "Android/5.10" as per TOR example
        def is_fake_os(os_str):
            if pd.isna(os_str): return 0
            if 'iOS/2' in str(os_str): return 1 # iOS 20+
            if 'Android/5.10' in str(os_str): return 1
            return 0
            
        self.df['is_fake_os'] = self.df['last_os_categorical'].apply(is_fake_os)

    def generate_graph_features(self):
        """Generates graph features based on 'direction'."""
        logger.info("Generating graph features...")
        
        # Build a graph where nodes are clients and directions
        # But 'direction' is just a hash.
        # We want features for the 'direction' (receiver).
        # Fan-in: how many unique clients sent to this direction
        
        # Calculate fan-in for each direction
        fan_in = self.df.groupby('direction')['cst_dim_id'].nunique().to_dict()
        self.df['direction_fan_in'] = self.df['direction'].map(fan_in)
        
        # Burst Receive: variance of time between incoming transfers
        # This requires grouping by direction and calculating stats on transdatetime
        # This is expensive, let's do a simplified version: count of txns to this direction
        txn_count = self.df.groupby('direction').size().to_dict()
        self.df['direction_txn_count'] = self.df['direction'].map(txn_count)

    def generate_time_delta_features(self):
        """Generates time delta features."""
        logger.info("Generating time delta features...")
        
        self.df['time_since_last_txn'] = self.df.groupby('cst_dim_id')['transdatetime'].diff().dt.total_seconds()
        self.df['time_since_last_txn'] = self.df['time_since_last_txn'].fillna(3600*24*30) # Fill NaNs with large value (first txn)

    def generate_gnn_features(self):
        """Generates graph embeddings using GraphEmbedderAgent."""
        logger.info("Generating GNN embeddings...")
        from src.agents.graph_embedder import GraphEmbedderAgent
        
        embedder = GraphEmbedderAgent(self.df)
        embedder.prepare_graph()
        embedder.train_embeddings(epochs=30) # Train for a bit longer
        emb_df = embedder.get_embeddings_df()
        
        # Merge embeddings back to main dataframe
        # We merge on cst_dim_id
        self.df = self.df.merge(emb_df, on='cst_dim_id', how='left')
        
        # Fill NaNs (for new clients not in graph training set, though here we train on current df)
        gnn_cols = [c for c in emb_df.columns if 'gnn_emb' in c]
        self.df[gnn_cols] = self.df[gnn_cols].fillna(0)

    def generate_anomaly_features(self):
        """Generates unsupervised anomaly scores using Isolation Forest."""
        logger.info("Generating anomaly features (Isolation Forest)...")
        from sklearn.ensemble import IsolationForest
        
        # Select features for anomaly detection
        # We use a subset of numerical features that likely capture abnormal behavior
        anomaly_cols = ['amount', 'count_txn_1h', 'time_since_last_txn']
        
        # Handle NaNs just in case (though we filled them)
        X_anomaly = self.df[anomaly_cols].fillna(0)
        
        # Fit Isolation Forest
        # contamination='auto' lets the model decide threshold
        # IMPORTANT: n_jobs=1 to avoid segfault on macOS ARM64
        iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=1)
        
        # We want the anomaly score. 
        # decision_function returns negative for outliers, positive for inliers.
        # We invert it so higher score = more anomalous.
        self.df['anomaly_score'] = -iso_forest.fit_predict(X_anomaly) # -1 for outlier, 1 for inlier. This is just labels.
        self.df['anomaly_score_raw'] = -iso_forest.decision_function(X_anomaly) # Higher = more anomalous


if __name__ == "__main__":
    # Test run
    from data_steward import DataStewardAgent
    
    steward = DataStewardAgent(
        behavior_path='datasets/behavior_patterns.csv',
        transactions_path='datasets/transactions.csv'
    )
    steward.load_data()
    steward.clean_data()
    merged_df = steward.merge_data()
    
    engineer = FeatureEngineerAgent(merged_df)
    features_df = engineer.generate_features()
    
    print(features_df[['cst_dim_id', 'transdatetime', 'count_txn_1h', 'high_risk_device', 'direction_fan_in']].head())
