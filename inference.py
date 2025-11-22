import argparse
import pandas as pd
import logging
from src.agents.data_steward import DataStewardAgent
from src.agents.feature_engineer import FeatureEngineerAgent
from src.agents.model_architect import ModelArchitectAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference(transactions_path, behavior_path, output_path='predictions.csv'):
    logger.info(f"Starting inference on {transactions_path}...")
    
    # 1. Load and Prepare Data
    # We reuse DataSteward to ensure same preprocessing logic
    steward = DataStewardAgent(
        behavior_path=behavior_path,
        transactions_path=transactions_path
    )
    steward.load_data()
    steward.clean_data()
    merged_df = steward.merge_data()
    
    # 2. Generate Features
    # We reuse FeatureEngineer to ensure same feature definitions
    engineer = FeatureEngineerAgent(merged_df)
    features_df = engineer.generate_features()
    
    # 3. Load Models and Predict
    architect = ModelArchitectAgent(features_df)
    architect.prepare_data()
    architect.load_models(path='models/')
    
    logger.info("Predicting...")
    probs = architect.predict_stacking_inference(features_df[architect.model_features])
    
    # 4. Save Results
    results = pd.DataFrame({
        'transdatetime': features_df['transdatetime'],
        'cst_dim_id': features_df['cst_dim_id'],
        'amount': features_df['amount'],
        'fraud_probability': probs,
        'is_fraud_prediction': (probs > 0.5).astype(int)
    })
    
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run fraud detection inference on new data.')
    parser.add_argument('--transactions', type=str, required=True, help='Path to new transactions CSV')
    parser.add_argument('--behavior', type=str, required=True, help='Path to behavior CSV (can be same as training or updated)')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output path for predictions')
    
    args = parser.parse_args()
    
    run_inference(args.transactions, args.behavior, args.output)
