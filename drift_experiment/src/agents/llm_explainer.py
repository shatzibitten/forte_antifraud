import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMExplainerAgent:
    """
    Agent that generates natural language explanations for suspicious transactions.
    
    It uses SHAP values to identify the most important features contributing to the
    fraud probability and constructs a human-readable narrative.
    """
    def __init__(self, model, df, features):
        self.model = model
        self.df = df.copy()
        self.features = features
        
        # Feature descriptions for better readability
        self.feature_descriptions = {
            'amount': 'Transaction Amount',
            'count_txn_1h': 'Number of transactions in the last hour',
            'time_since_last_txn': 'Time elapsed since the last transaction',
            'anomaly_score': 'Anomaly Score (Isolation Forest)',
            'gnn_emb_0': 'Graph Embedding (Network Pattern)',
            'night_transaction': 'Transaction occurred at night',
            'diff_device_hash': 'Device change detected',
            'diff_ip_hash': 'IP address change detected',
            'velocity_6h': 'Transaction velocity (6 hours)',
            'interaction_night_amount': 'High amount during night time'
        }

    def explain_transaction(self, transaction_idx, shap_values, top_k=3):
        """
        Generates a natural language explanation for a specific transaction.
        
        Args:
            transaction_idx: Index of the transaction in the dataframe.
            shap_values: SHAP values for this transaction.
            top_k: Number of top features to include in the explanation.
            
        Returns:
            A string containing the explanation.
        """
        row = self.df.iloc[transaction_idx]
        shap_vals = shap_values[transaction_idx]
        
        # Create a DataFrame of features and their SHAP values
        explanation_df = pd.DataFrame({
            'feature': self.features,
            'value': row[self.features].values,
            'shap_value': shap_vals
        })
        
        # Sort by SHAP value (descending) to find features pushing towards fraud
        top_features = explanation_df.sort_values('shap_value', ascending=False).head(top_k)
        
        # Construct the narrative
        explanation = f"⚠️ **Suspicious Transaction Alert** (ID: {transaction_idx})\n"
        explanation += f"**Risk Probability:** {row.get('pred_prob', 'N/A'):.2%}\n\n"
        explanation += "**Why is this suspicious?**\n"
        
        for _, feat_row in top_features.iterrows():
            feature_name = feat_row['feature']
            readable_name = self.feature_descriptions.get(feature_name, feature_name)
            value = feat_row['value']
            shap_impact = feat_row['shap_value']
            
            if shap_impact > 0:
                explanation += f"- **{readable_name}**: Value is `{value:.2f}`. This increases the risk significantly.\n"
            
        explanation += "\n**Recommendation:** Verify this transaction with the customer immediately."
        
        return explanation

    def explain_top_fraud_cases(self, n_cases=5):
        """Explains the top N most suspicious transactions in the dataset."""
        logger.info(f"Generating explanations for top {n_cases} fraud cases...")
        
        # Ensure we have predictions
        if 'pred_prob' not in self.df.columns:
            self.df['pred_prob'] = self.model.predict_proba(self.df[self.features])[:, 1]
            
        # Get indices of top fraud cases
        top_indices = self.df.sort_values('pred_prob', ascending=False).head(n_cases).index
        
        # We need SHAP values for these specific rows
        # Note: In a real scenario, we'd pass pre-calculated SHAP values
        # Here we calculate them on the fly for the top cases
        import shap
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.df.loc[top_indices, self.features])
        
        explanations = []
        for i, idx in enumerate(top_indices):
            # shap_values is an array, so we access by 0-based index relative to the subset
            exp = self.explain_transaction(i, shap_values, top_k=3) # Pass relative index i, but use original idx for display
            # Fix: explain_transaction expects index into the passed shap_values array
            # But it also needs access to the original row.
            # Let's refactor explain_transaction slightly to take the row and shap array directly.
            
            # Actually, let's just inline the logic here for simplicity or fix the method
            row = self.df.loc[idx]
            shap_vals = shap_values[i]
            
            explanation_df = pd.DataFrame({
                'feature': self.features,
                'value': row[self.features].values,
                'shap_value': shap_vals
            })
            
            top_features = explanation_df.sort_values('shap_value', ascending=False).head(3)
            
            explanation = f"### Case #{i+1} (Transaction ID: {idx})\n"
            explanation += f"**Risk Score:** {row['pred_prob']:.4f}\n"
            explanation += "**Key Risk Factors:**\n"
            
            for _, feat_row in top_features.iterrows():
                feature_name = feat_row['feature']
                readable_name = self.feature_descriptions.get(feature_name, feature_name)
                value = feat_row['value']
                
                explanation += f"- **{readable_name}**: {value:.2f}\n"
                
            explanations.append(explanation)
            
        return "\n\n".join(explanations)
