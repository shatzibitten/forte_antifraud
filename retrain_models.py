# IMPORTANT: Import this FIRST to prevent segfault on macOS ARM64
import src.fix_segfault

import logging
import pandas as pd
from src.agents.data_steward import DataStewardAgent
from src.agents.feature_engineer import FeatureEngineerAgent
from src.agents.model_architect import ModelArchitectAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Model Retraining...")
    
    # 1. Data Steward
    steward = DataStewardAgent(
        behavior_path='datasets/behavior_patterns.csv',
        transactions_path='datasets/transactions.csv'
    )
    steward.load_data()
    steward.clean_data()
    merged_df = steward.merge_data()
    
    # 2. Feature Engineer
    engineer = FeatureEngineerAgent(merged_df)
    features_df = engineer.generate_features()
    
    # 3. Model Architect
    architect = ModelArchitectAgent(features_df)
    architect.prepare_data()
    
    # Train base models
    architect.train_final_model(use_optuna=False) # Trains CatBoost
    architect.train_lightgbm(use_optuna=False)
    architect.train_xgboost(use_optuna=False)
    
    # Train stacking
    architect.train_stacking()
    
    # Save models
    architect.save_models(path='models/')
    logger.info("Retraining complete. Models saved.")

if __name__ == "__main__":
    main()
