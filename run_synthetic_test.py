# IMPORTANT: Import this FIRST to prevent segfault on macOS ARM64
import src.fix_segfault

import logging
import pandas as pd
import numpy as np
from src.agents.data_steward import DataStewardAgent
from src.agents.feature_engineer import FeatureEngineerAgent
from src.agents.model_architect import ModelArchitectAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Synthetic Holdout Test...")

    # Paths to synthetic data
    transactions_path = 'datasets/Test/transactions_synthetic_rules.csv'
    behavior_path = 'datasets/Test/behavior_patterns_synthetic_rules.csv'
    
    # 1. Load and Prepare Data
    # Note: behavior file has 2 header lines (skiprows=2 is default in DataStewardAgent)
    # transactions file has 1 header line (skiprows=1 is default in DataStewardAgent)
    steward = DataStewardAgent(
        behavior_path=behavior_path,
        transactions_path=transactions_path
    )
    steward.load_data()
    steward.clean_data()
    merged_df = steward.merge_data()
    
    # 2. Generate Features
    engineer = FeatureEngineerAgent(merged_df)
    features_df = engineer.generate_features()
    
    # 3. Load Models and Predict
    architect = ModelArchitectAgent(features_df)
    architect.prepare_data()
    architect.load_models(path='models/')
    
    logger.info("Predicting on synthetic data...")
    probs = architect.predict_stacking_inference(architect.df)
    
    # 4. Calculate Business Metrics
    # We need 'target' and 'amount' for this.
    # The synthetic data HAS a target column.
    
    results = pd.DataFrame({
        'transdatetime': features_df['transdatetime'],
        'cst_dim_id': features_df['cst_dim_id'],
        'amount': features_df['amount'],
        'target': features_df['target'], # Ground truth
        'fraud_probability': probs,
        'is_fraud_prediction': (probs > 0.5).astype(int)
    })
    
    # Calculate metrics
    # Saved Money: Fraud caught (TP) * Amount
    # Lost Money: Fraud missed (FN) * Amount
    
    tp_mask = (results['target'] == 1) & (results['is_fraud_prediction'] == 1)
    fn_mask = (results['target'] == 1) & (results['is_fraud_prediction'] == 0)
    fp_mask = (results['target'] == 0) & (results['is_fraud_prediction'] == 1)
    
    saved_money = results.loc[tp_mask, 'amount'].sum()
    lost_money = results.loc[fn_mask, 'amount'].sum()
    false_positive_loss = results.loc[fp_mask, 'amount'].sum() # Potential friction cost, usually not counted as direct loss but good to know
    
    logger.info(f"\n{'='*60}")
    logger.info("SYNTHETIC HOLDOUT TEST RESULTS:")
    logger.info(f"{'='*60}")
    logger.info(f"Total Transactions: {len(results)}")
    logger.info(f"Total Frauds (Ground Truth): {results['target'].sum()}")
    logger.info(f"Predicted Frauds: {results['is_fraud_prediction'].sum()}")
    logger.info(f"{'-'*30}")
    logger.info(f"Saved Money (TP Amount): {saved_money:,.2f} KZT")
    logger.info(f"Lost Money (FN Amount): {lost_money:,.2f} KZT")
    logger.info(f"False Positive Amount: {false_positive_loss:,.2f} KZT")
    logger.info(f"{'='*60}\n")
    
    # Save results
    output_path = 'predictions_synthetic_test.csv'
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
