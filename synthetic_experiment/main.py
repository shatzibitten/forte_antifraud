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
    logger.info("Starting Synthetic Data Experiment...")
    
    # ==========================================
    # PHASE 1: Train on FULL Original Dataset
    # ==========================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Training on FULL Original Dataset (No Holdout)")
    logger.info("="*60)
    
    # 1. Data Steward (Original)
    steward = DataStewardAgent(
        behavior_path='datasets/behavior_patterns.csv',
        transactions_path='datasets/transactions.csv'
    )
    steward.load_data()
    steward.clean_data()
    merged_df = steward.merge_data()
    
    # 2. Feature Engineer (Original)
    engineer = FeatureEngineerAgent(merged_df)
    features_df = engineer.generate_features()
    
    logger.info(f"Full Training Data: {len(features_df)} transactions")
    logger.info(f"Training Period: {features_df['transdatetime'].min()} to {features_df['transdatetime'].max()}")
    
    # 3. Model Architect (Train on ALL data)
    architect = ModelArchitectAgent(features_df)
    architect.prepare_data()
    
    # Train individual models (Optuna disabled for speed/stability in this test)
    logger.info("Training models on full dataset...")
    architect.train_final_model(use_optuna=False) 
    architect.train_lightgbm(use_optuna=False)    
    architect.train_xgboost(use_optuna=False)     
    
    # Train Stacking Meta-Model
    architect.train_stacking()
    architect.save_models()
    
    # ==========================================
    # PHASE 2: Evaluate on Synthetic Data
    # ==========================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Evaluating on Synthetic Data")
    logger.info("="*60)
    
    # 1. Data Steward (Synthetic)
    # Note: We use the same class but point to synthetic files
    synth_steward = DataStewardAgent(
        behavior_path='datasets/Test/behavior_patterns_synthetic_rules.csv',
        transactions_path='datasets/Test/transactions_synthetic_rules.csv'
    )
    synth_steward.load_data()
    synth_steward.clean_data()
    synth_merged_df = synth_steward.merge_data()
    
    # 2. Feature Engineer (Synthetic)
    synth_engineer = FeatureEngineerAgent(synth_merged_df)
    synth_features_df = synth_engineer.generate_features()
    
    logger.info(f"Synthetic Test Data: {len(synth_features_df)} transactions")
    
    # 3. Prepare Synthetic Data for Inference
    # We create a new architect just to use its prepare_data method for cleaning
    synth_architect = ModelArchitectAgent(synth_features_df)
    synth_architect.prepare_data()
    
    # 4. Evaluation
    class StackingWrapper:
        def __init__(self, architect):
            self.architect = architect
        def predict_proba(self, X):
            probs = self.architect.predict_stacking(X)
            return np.vstack([1-probs, probs]).T
    
    stacking_model = StackingWrapper(architect)
    
    # Metrics on Synthetic Data
    logger.info(f"\n{'='*60}")
    logger.info("SYNTHETIC DATA METRICS:")
    logger.info(f"{'='*60}")
    
    # Use the TRAINED architect to predict on SYNTHETIC data
    # We pass synth_architect.df but use architect.model_features to ensure column alignment
    synth_auditor = AuditorAgent(stacking_model, synth_architect.df, architect.model_features)
    synth_metrics = synth_auditor.check_business_metrics()
    
    # Drift Analysis (Original vs Synthetic)
    logger.info(f"\n{'='*60}")
    logger.info("DRIFT ANALYSIS (Original vs Synthetic):")
    logger.info(f"{'='*60}")
    # Use auditor instance from original to call analyze_drift
    # We create a dummy auditor just for the method
    auditor_tool = AuditorAgent(None, architect.df, architect.model_features)
    auditor_tool.analyze_drift(architect.df, synth_architect.df)
    
    # LLM Explainer on Synthetic Data
    logger.info("\n" + "="*60)
    logger.info("LLM EXPLAINER (Top 3 Suspicious Synthetic Cases)")
    logger.info("="*60)
    
    llm_agent = LLMExplainerAgent(architect.catboost_model, synth_architect.df, architect.model_features)
    explanations = llm_agent.explain_top_fraud_cases(n_cases=3)
    logger.info("\n" + explanations)
    
    return synth_metrics

if __name__ == "__main__":
    main()
