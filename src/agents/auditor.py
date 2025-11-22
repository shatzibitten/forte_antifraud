import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AuditorAgent:
    def __init__(self, model, df, features):
        self.model = model
        self.df = df.copy()
        self.features = features

    def explain_shap(self, output_path='shap_summary.png'):
        """Generates SHAP summary plot."""
        logger.info("Generating SHAP explanation...")
        
        # Create explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values for a sample (to save time)
        sample_df = self.df[self.features].sample(n=min(1000, len(self.df)), random_state=42)
        shap_values = explainer.shap_values(sample_df)
        
        # Plot summary
        plt.figure()
        shap.summary_plot(shap_values, sample_df, show=False)
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"SHAP summary saved to {output_path}")
        
        return shap_values

    def check_business_metrics(self, threshold=0.5):
        """Calculates business metrics."""
        logger.info("Calculating business metrics...")
        
        preds = self.model.predict_proba(self.df[self.features])[:, 1]
        self.df['pred_prob'] = preds
        self.df['pred_class'] = (preds > threshold).astype(int)
        
        # Calculate saved money (True Positives * Amount)
        saved_money = self.df[(self.df['target'] == 1) & (self.df['pred_class'] == 1)]['amount'].sum()
        
        # Calculate lost money (False Negatives * Amount)
        lost_money = self.df[(self.df['target'] == 1) & (self.df['pred_class'] == 0)]['amount'].sum()
        
        # Calculate blocked legit money (False Positives * Amount) - Friction cost
        blocked_legit = self.df[(self.df['target'] == 0) & (self.df['pred_class'] == 1)]['amount'].sum()
        
        logger.info(f"Saved Money: {saved_money:,.2f}")
        logger.info(f"Lost Money: {lost_money:,.2f}")
        logger.info(f"Blocked Legit Money: {blocked_legit:,.2f}")
        
        return {
            'saved_money': saved_money,
            'lost_money': lost_money,
            'blocked_legit': blocked_legit
        }

if __name__ == "__main__":
    # Test run
    from data_steward import DataStewardAgent
    from feature_engineer import FeatureEngineerAgent
    from model_architect import ModelArchitectAgent
    
    # 1. Data
    steward = DataStewardAgent(
        behavior_path='datasets/behavior_patterns.csv',
        transactions_path='datasets/transactions.csv'
    )
    steward.load_data()
    steward.clean_data()
    merged_df = steward.merge_data()
    
    # 2. Features
    engineer = FeatureEngineerAgent(merged_df)
    features_df = engineer.generate_features()
    
    # 3. Model
    architect = ModelArchitectAgent(features_df)
    architect.prepare_data()
    model = architect.train_final_model()
    
    # 4. Audit
    # Use architect.df because it has been cleaned (NaNs handled, strings converted)
    auditor = AuditorAgent(model, architect.df, architect.model_features)
    auditor.explain_shap()
    auditor.check_business_metrics()
