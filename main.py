import logging
import numpy as np
from src.agents.data_steward import DataStewardAgent
from src.agents.feature_engineer import FeatureEngineerAgent
from src.agents.model_architect import ModelArchitectAgent
from src.agents.auditor import AuditorAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting ForteBank AI Hackathon Pipeline...")
    
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
    
    # Train individual models for final prediction usage
    architect.train_final_model() # CatBoost
    architect.train_lightgbm()    # LightGBM
    
    # Train Stacking Meta-Model
    architect.train_stacking()
    
    # 4. Auditor
    # We need to wrap the stacking prediction in a way Auditor accepts (it expects a model with predict_proba)
    class StackingWrapper:
        def __init__(self, architect):
            self.architect = architect
        def predict_proba(self, X):
            # Auditor passes X which might be DataFrame
            # predict_stacking returns 1D array of probs for class 1
            probs = self.architect.predict_stacking(X)
            # sklearn expects [prob_0, prob_1]
            return np.vstack([1-probs, probs]).T
            
    stacking_model = StackingWrapper(architect)
    
    auditor = AuditorAgent(stacking_model, architect.df, architect.model_features)
    
    # SHAP might fail with StackingWrapper because it's not a Tree model.
    # For SHAP, we'll use the CatBoost model as the primary explainer (most interpretable)
    logger.info("Using CatBoost for SHAP explanation...")
    auditor_shap = AuditorAgent(architect.catboost_model, architect.df, architect.model_features)
    auditor_shap.explain_shap()
    
    # Business Metrics using Stacking
    logger.info("Calculating Business Metrics for Stacking Ensemble...")
    auditor.check_business_metrics()

if __name__ == "__main__":
    main()
