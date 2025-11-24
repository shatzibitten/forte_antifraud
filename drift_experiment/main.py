# IMPORTANT: Import this FIRST to prevent segfault on macOS ARM64
import src.fix_segfault

import logging
import numpy as np
import pandas as pd
from src.agents.data_steward import DataStewardAgent
from src.agents.feature_engineer import FeatureEngineerAgent
from src.agents.model_architect import ModelArchitectAgent
from src.agents.auditor import AuditorAgent
from src.agents.llm_explainer import LLMExplainerAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Pseudo-Labeling Experiment...")
    
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
    
    # **Holdout Test Set: Reserve last 15% of data by time**
    features_df = features_df.sort_values('transdatetime')
    holdout_size = int(len(features_df) * 0.15)
    
    train_df = features_df.iloc[:-holdout_size].copy()
    holdout_df = features_df.iloc[-holdout_size:].copy()
    
    logger.info(f"\n{'='*60}")
    logger.info("DATA SPLIT:")
    logger.info(f"Train set: {len(train_df)}")
    logger.info(f"Holdout set: {len(holdout_df)}")
    logger.info(f"{'='*60}\n")
    
    # 3. Initial Training
    logger.info("Step 1: Initial Training...")
    architect = ModelArchitectAgent(train_df)
    architect.prepare_data()
    
    architect.train_final_model(use_optuna=False)
    architect.train_lightgbm(use_optuna=False)
    architect.train_xgboost(use_optuna=False)
    architect.train_stacking()
    
    # 4. Pseudo-Labeling
    logger.info("\n" + "="*60)
    logger.info("Step 2: Generating Pseudo-Labels...")
    logger.info("="*60)
    
    # Predict on Holdout
    holdout_architect = ModelArchitectAgent(holdout_df)
    holdout_architect.prepare_data()
    
    # Get predictions from the stacking model
    probs = architect.predict_stacking(holdout_architect.df[architect.model_features])
    
    holdout_df['pred_prob'] = probs
    
    # Select high-confidence samples
    # Confidence > 0.95 (Fraud) or < 0.01 (Legit)
    high_conf_fraud = holdout_df[holdout_df['pred_prob'] > 0.95].copy()
    high_conf_legit = holdout_df[holdout_df['pred_prob'] < 0.01].copy()
    
    # Assign pseudo-labels
    high_conf_fraud['target'] = 1
    high_conf_legit['target'] = 0
    
    logger.info(f"Found {len(high_conf_fraud)} high-confidence FRAUD pseudo-labels")
    logger.info(f"Found {len(high_conf_legit)} high-confidence LEGIT pseudo-labels")
    
    # Combine with original train
    pseudo_train_df = pd.concat([train_df, high_conf_fraud, high_conf_legit], axis=0)
    
    # 5. Retraining with Pseudo-Labels
    logger.info("\n" + "="*60)
    logger.info("Step 3: Retraining with Pseudo-Labels...")
    logger.info("="*60)
    
    architect_pseudo = ModelArchitectAgent(pseudo_train_df)
    architect_pseudo.prepare_data()
    
    architect_pseudo.train_final_model(use_optuna=False)
    architect_pseudo.train_lightgbm(use_optuna=False)
    architect_pseudo.train_xgboost(use_optuna=False)
    architect_pseudo.train_stacking()
    
    # 6. Final Evaluation on Holdout
    class StackingWrapper:
        def __init__(self, architect):
            self.architect = architect
        def predict_proba(self, X):
            probs = self.architect.predict_stacking(X)
            return np.vstack([1-probs, probs]).T
            
    stacking_model = StackingWrapper(architect_pseudo)
    
    logger.info(f"\n{'='*60}")
    logger.info("FINAL METRICS (After Pseudo-Labeling):")
    logger.info(f"{'='*60}")
    
    holdout_auditor = AuditorAgent(stacking_model, holdout_architect.df, architect_pseudo.model_features)
    holdout_metrics = holdout_auditor.check_business_metrics()
    
    # logger.info(f"Train Set Saved: {train_metrics['saved_money']:,.2f} KZT") # Undefined
    logger.info(f"Holdout Set Saved: {holdout_metrics['saved_money']:,.2f} KZT")
    logger.info(f"Holdout Set Lost: {holdout_metrics['lost_money']:,.2f} KZT")
    logger.info(f"{'='*60}\n")
    
    return {
        'train_metrics': train_metrics,
        'holdout_metrics': holdout_metrics
    }

if __name__ == "__main__":
    main()
