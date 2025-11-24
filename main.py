# IMPORTANT: Import this FIRST to prevent segfault on macOS ARM64
import src.fix_segfault

import logging
import yaml
import numpy as np
import pandas as pd
from src.agents.data_steward import DataStewardAgent
from src.agents.feature_engineer import FeatureEngineerAgent
from src.agents.model_architect import ModelArchitectAgent
from src.agents.auditor import AuditorAgent
from src.agents.llm_explainer import LLMExplainerAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    logger.info("Starting ForteBank AI Hackathon Pipeline...")
    
    # Load Config
    config = load_config()
    
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
    logger.info(f"Train set: {len(train_df)} transactions ({train_df['transdatetime'].min()} to {train_df['transdatetime'].max()})")
    logger.info(f"Holdout set: {len(holdout_df)} transactions ({holdout_df['transdatetime'].min()} to {holdout_df['transdatetime'].max()})")
    logger.info(f"Train fraud rate: {train_df['target'].mean():.4f}")
    logger.info(f"Holdout fraud rate: {holdout_df['target'].mean():.4f}")
    logger.info(f"{'='*60}\n")
    
    # 3. Model Architect (train on train_df with TimeSeriesSplit CV)
    architect = ModelArchitectAgent(train_df)
    architect.prepare_data()
    
    # Train individual models with Optuna optimization
    use_optuna = config.get('model', {}).get('use_optuna', False)
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING MODELS (Optuna: {use_optuna})")
    logger.info("="*60 + "\n")
    
    architect.train_final_model(use_optuna=use_optuna) 
    architect.train_lightgbm(use_optuna=use_optuna)    
    architect.train_xgboost(use_optuna=use_optuna)     
    
    # Train Stacking Meta-Model (uses TimeSeriesSplit internally)
    architect.train_stacking()
    
    # Save models for future inference
    architect.save_models()
    
    # 4. Evaluation
    class StackingWrapper:
        def __init__(self, architect):
            self.architect = architect
        def predict_proba(self, X):
            probs = self.architect.predict_stacking(X)
            return np.vstack([1-probs, probs]).T
    
    stacking_model = StackingWrapper(architect)
    
    # 4a. SHAP explanation on train data
    logger.info("Generating SHAP explanations...")
    auditor_shap = AuditorAgent(architect.catboost_model, architect.df, architect.model_features)
    auditor_shap.explain_shap()
    
    threshold = config.get('fraud_threshold', 0.5)
    logger.info(f"Using Fraud Threshold from config: {threshold}")

    # 4b. Metrics on TRAIN set (for reference)
    logger.info(f"\n{'='*60}")
    logger.info("TRAIN SET METRICS (reference only):")
    logger.info(f"{'='*60}")
    train_auditor = AuditorAgent(stacking_model, architect.df, architect.model_features)
    train_metrics = train_auditor.check_business_metrics(threshold=threshold)
    
    # 4c. Metrics on HOLDOUT TEST set (FINAL evaluation)
    logger.info(f"\n{'='*60}")
    logger.info("HOLDOUT TEST SET METRICS (final evaluation):")
    logger.info(f"{'='*60}")
    
    # Prepare holdout data with same transformations
    holdout_architect = ModelArchitectAgent(holdout_df)
    holdout_architect.prepare_data()
    
    # DRIFT ANALYSIS
    # Use auditor_shap instance to call analyze_drift
    auditor_shap.analyze_drift(architect.df, holdout_architect.df)
    
    # LLM EXPLAINER AGENT
    logger.info("\n" + "="*60)
    logger.info("LLM EXPLAINER AGENT (Top 3 Suspicious Cases in Holdout)")
    logger.info("="*60)
    
    # We use the CatBoost model for explanations as it's tree-based and works with TreeExplainer
    llm_agent = LLMExplainerAgent(architect.catboost_model, holdout_architect.df, architect.model_features)
    explanations = llm_agent.explain_top_fraud_cases(n_cases=3)
    logger.info("\n" + explanations)
    logger.info("="*60 + "\n")
    
    holdout_auditor = AuditorAgent(stacking_model, holdout_architect.df, architect.model_features)
    holdout_metrics = holdout_auditor.check_business_metrics(threshold=threshold)
    
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY:")
    logger.info(f"{'='*60}")
    logger.info(f"Train Set Saved: {train_metrics['saved_money']:,.2f} KZT")
    logger.info(f"Holdout Set Saved: {holdout_metrics['saved_money']:,.2f} KZT")
    logger.info(f"Holdout Set Lost: {holdout_metrics['lost_money']:,.2f} KZT")
    logger.info(f"{'='*60}\n")
    
    return {
        'train_metrics': train_metrics,
        'holdout_metrics': holdout_metrics
    }

if __name__ == "__main__":
    main()
