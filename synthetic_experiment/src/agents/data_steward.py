import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataStewardAgent:
    def __init__(self, behavior_path, transactions_path, skiprows_behavior=2, skiprows_transactions=1):
        self.behavior_path = behavior_path
        self.transactions_path = transactions_path
        self.skiprows_behavior = skiprows_behavior
        self.skiprows_transactions = skiprows_transactions
        self.behavior_df = None
        self.transactions_df = None
        self.merged_df = None

    def load_data(self):
        """Loads data from CSV files."""
        logger.info("Loading data...")
        try:
            # Behavior patterns has 2 lines of description/header garbage
            self.behavior_df = pd.read_csv(self.behavior_path, skiprows=self.skiprows_behavior, sep=';', quotechar="'", encoding='cp1251')
            # Transactions has 1 line of description
            self.transactions_df = pd.read_csv(self.transactions_path, skiprows=self.skiprows_transactions, sep=';', quotechar="'", encoding='cp1251')
            
            logger.info(f"Loaded behavior data: {self.behavior_df.shape}")
            logger.info(f"Loaded transactions data: {self.transactions_df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def clean_data(self):
        """Cleans and preprocesses the data."""
        logger.info("Cleaning data...")
        
        # --- Transactions Cleaning ---
        # Convert dates
        self.transactions_df['transdatetime'] = pd.to_datetime(self.transactions_df['transdatetime'])
        self.transactions_df['transdate'] = pd.to_datetime(self.transactions_df['transdate'])
        
        # Ensure amount is float
        self.transactions_df['amount'] = self.transactions_df['amount'].astype(float)
        
        # Convert cst_dim_id to int64 (drop NaNs first if any)
        self.transactions_df = self.transactions_df.dropna(subset=['cst_dim_id'])
        self.transactions_df['cst_dim_id'] = self.transactions_df['cst_dim_id'].astype(np.int64)
        
        # Sort by time for merge_asof
        self.transactions_df = self.transactions_df.sort_values('transdatetime')

        # --- Behavior Cleaning ---
        # Convert dates
        self.behavior_df['transdate'] = pd.to_datetime(self.behavior_df['transdate'])
        
        # Convert cst_dim_id to int64
        self.behavior_df = self.behavior_df.dropna(subset=['cst_dim_id'])
        self.behavior_df['cst_dim_id'] = self.behavior_df['cst_dim_id'].astype(np.int64)
        
        # Handle missing values in behavior data
        # TOR 2.1.3: Create separate flags for missing values instead of mean imputation
        cols_with_nans = ['zscore_avg_login_interval_7d', 'burstiness_login_interval', 'fano_factor_login_interval']
        for col in cols_with_nans:
            if col in self.behavior_df.columns:
                self.behavior_df[f'{col}_is_missing'] = self.behavior_df[col].isna().astype(int)
                self.behavior_df[col] = self.behavior_df[col].fillna(0) # Fill with 0 after flagging
        
        # Sort by time (transdate) for merge_asof
        self.behavior_df = self.behavior_df.sort_values('transdate')

    def merge_data(self):
        """Merges transactions with behavioral data using merge_asof."""
        logger.info("Merging data...")
        logger.info(f"Transactions dtypes: transdatetime={self.transactions_df['transdatetime'].dtype}, cst_dim_id={self.transactions_df['cst_dim_id'].dtype}")
        logger.info(f"Behavior dtypes: transdate={self.behavior_df['transdate'].dtype}, cst_dim_id={self.behavior_df['cst_dim_id'].dtype}")
        
        # merge_asof requires sorted keys
        # We merge transactions (left) with behavior (right) on 'transdatetime' vs 'transdate'
        # and match on 'cst_dim_id'
        # direction='backward' ensures we only see PAST behavior
        
        self.merged_df = pd.merge_asof(
            self.transactions_df,
            self.behavior_df,
            left_on='transdatetime',
            right_on='transdate',
            by='cst_dim_id',
            direction='backward',
            suffixes=('_txn', '_beh')
        )
        
        logger.info(f"Merged data shape: {self.merged_df.shape}")
        return self.merged_df

if __name__ == "__main__":
    # Test run
    agent = DataStewardAgent(
        behavior_path='datasets/behavior_patterns.csv',
        transactions_path='datasets/transactions.csv'
    )
    agent.load_data()
    agent.clean_data()
    df = agent.merge_data()
    print(df.head())
    print(df.columns)
