# IMPORTANT: Import this FIRST to prevent segfault on macOS ARM64
import src.fix_segfault

import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.agents.data_steward import DataStewardAgent
from src.agents.feature_engineer import FeatureEngineerAgent
from src.agents.model_architect import ModelArchitectAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_adversarial_validation():
    logger.info("Starting Adversarial Validation...")
    
    # 1. Load and Prepare Data
    steward = DataStewardAgent(
        behavior_path='datasets/behavior_patterns.csv',
        transactions_path='datasets/transactions.csv'
    )
    steward.load_data()
    steward.clean_data()
    merged_df = steward.merge_data()
    
    engineer = FeatureEngineerAgent(merged_df)
    features_df = engineer.generate_features()
    
    # 2. Split into Train and Holdout (same as before)
    features_df = features_df.sort_values('transdatetime')
    holdout_size = int(len(features_df) * 0.15)
    
    train_df = features_df.iloc[:-holdout_size].copy()
    holdout_df = features_df.iloc[-holdout_size:].copy()
    
    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Holdout size: {len(holdout_df)}")
    
    # 3. Prepare Data for Adversarial Validation
    # We want to distinguish between Train (0) and Holdout (1)
    train_df['is_holdout'] = 0
    holdout_df['is_holdout'] = 1
    
    adv_df = pd.concat([train_df, holdout_df], axis=0)
    
    # Use the same features as the main model
    architect = ModelArchitectAgent(train_df)
    architect.prepare_data() # To get the feature list
    features = architect.model_features
    
    # Remove features that are explicitly time-based or target-related
    features = [f for f in features if f not in ['target', 'is_holdout', 'transdatetime', 'transdate']]
    
    logger.info(f"Using {len(features)} features for adversarial validation.")
    
    X = adv_df[features]
    y = adv_df['is_holdout']
    
    # Handle NaNs and Categoricals
    # Simple filling for this check
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str)
            X[col] = X[col].astype('category')
        else:
            X[col] = X[col].fillna(0)
            
    # 4. Train Classifier
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    importances = pd.DataFrame(index=features)
    importances['total_gain'] = 0
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        scores.append(score)
        
        importances['total_gain'] += model.feature_importances_
        
        logger.info(f"Fold {fold+1} AUC: {score:.4f}")
        
    mean_auc = np.mean(scores)
    logger.info(f"\nMean Adversarial AUC: {mean_auc:.4f}")
    
    if mean_auc > 0.7:
        logger.warning("High Adversarial AUC indicates significant drift! The model can easily distinguish Train from Holdout.")
    else:
        logger.info("Low Adversarial AUC indicates Train and Holdout are similar.")
        
    # 5. Analyze Feature Importance
    importances['total_gain'] /= 5
    top_drifting = importances.sort_values('total_gain', ascending=False).head(10)
    
    logger.info("\nTop 10 Drifting Features (Most important for distinguishing Train vs Holdout):")
    logger.info(top_drifting)
    
    return top_drifting.index.tolist()

if __name__ == "__main__":
    run_adversarial_validation()
