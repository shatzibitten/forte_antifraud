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

    def optimize_threshold(self):
        """Finds optimal threshold by testing different values."""
        logger.info("Optimizing threshold for maximum profit...")
        
        # Get predictions
        preds = self.model.predict_proba(self.df[self.features])[:, 1]
        
        # Test thresholds from 0.1 to 0.9
        thresholds = np.arange(0.05, 0.95, 0.05)
        results = []
        
        for threshold in thresholds:
            pred_class = (preds > threshold).astype(int)
            
            # Calculate metrics
            saved = self.df[(self.df['target'] == 1) & (pred_class == 1)]['amount'].sum()
            lost = self.df[(self.df['target'] == 1) & (pred_class == 0)]['amount'].sum()
            blocked = self.df[(self.df['target'] == 0) & (pred_class == 1)]['amount'].sum()
            
            # Net profit: saved money - lost fraud - friction cost
            net_profit = saved - lost - blocked
            
            # Recall and Precision
            tp = ((self.df['target'] == 1) & (pred_class == 1)).sum()
            fn = ((self.df['target'] == 1) & (pred_class == 0)).sum()
            fp = ((self.df['target'] == 0) & (pred_class == 1)).sum()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'saved_money': saved,
                'lost_money': lost,
                'blocked_legit': blocked,
                'net_profit': net_profit,
                'recall': recall,
                'precision': precision
            })
        
        # Find best threshold by net profit
        results_df = pd.DataFrame(results)
        best_idx = results_df['net_profit'].idxmax()
        best = results_df.loc[best_idx]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMAL THRESHOLD: {best['threshold']:.2f}")
        logger.info(f"Saved Money: {best['saved_money']:,.2f}")
        logger.info(f"Lost Money: {best['lost_money']:,.2f}")
        logger.info(f"Blocked Legit: {best['blocked_legit']:,.2f}")
        logger.info(f"Net Profit: {best['net_profit']:,.2f}")
        logger.info(f"Recall: {best['recall']:.4f}")
        logger.info(f"Precision: {best['precision']:.4f}")
        logger.info(f"{'='*60}\n")
        
        return best['threshold'], results_df

    def analyze_drift(self, train_df, holdout_df):
        """Analyzes feature drift between train and holdout sets."""
        logger.info("Analyzing data drift...")
        
        drift_report = []
        
        for feature in self.features:
            if feature not in train_df.columns or feature not in holdout_df.columns:
                continue
                
            # Check if numerical
            if pd.api.types.is_numeric_dtype(train_df[feature]):
                mean_train = train_df[feature].mean()
                mean_holdout = holdout_df[feature].mean()
                std_train = train_df[feature].std()
                
                # Standardized difference in means
                if std_train > 0:
                    drift_score = abs(mean_train - mean_holdout) / std_train
                else:
                    drift_score = 0.0
                    
                drift_report.append({
                    'feature': feature,
                    'type': 'numerical',
                    'train_mean': mean_train,
                    'holdout_mean': mean_holdout,
                    'drift_score': drift_score
                })
            else:
                # Categorical drift (simple Jaccard or similar)
                # For simplicity, we'll skip detailed categorical drift here
                pass
                
        drift_df = pd.DataFrame(drift_report)
        if not drift_df.empty:
            drift_df = drift_df.sort_values('drift_score', ascending=False)
            
            logger.info(f"\n{'='*60}")
            logger.info("TOP 10 DRIFTING FEATURES:")
            logger.info(f"{'='*60}")
            for i, row in drift_df.head(10).iterrows():
                logger.info(f"{row['feature']}: score={row['drift_score']:.4f} (Train: {row['train_mean']:.2f} -> Holdout: {row['holdout_mean']:.2f})")
            logger.info(f"{'='*60}\n")
            
        return drift_df

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
