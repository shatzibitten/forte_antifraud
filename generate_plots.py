import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.calibration import calibration_curve
from src.agents.data_steward import DataStewardAgent
from src.agents.feature_engineer import FeatureEngineerAgent
from src.agents.model_architect import ModelArchitectAgent
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def generate_plots():
    logger.info("Loading data...")
    steward = DataStewardAgent(
        behavior_path='datasets/behavior_patterns.csv',
        transactions_path='datasets/transactions.csv'
    )
    steward.load_data()
    steward.clean_data()
    merged_df = steward.merge_data()
    
    logger.info("Generating features...")
    engineer = FeatureEngineerAgent(merged_df)
    features_df = engineer.generate_features()
    
    # Split
    features_df = features_df.sort_values('transdatetime')
    holdout_size = int(len(features_df) * 0.15)
    train_df = features_df.iloc[:-holdout_size].copy()
    holdout_df = features_df.iloc[-holdout_size:].copy()
    
    # Load Model
    logger.info("Loading models...")
    architect = ModelArchitectAgent(train_df)
    architect.load_models()
    
    # Prepare Holdout for Inference
    holdout_architect = ModelArchitectAgent(holdout_df)
    holdout_architect.prepare_data()
    
    # Get Predictions (Uncalibrated vs Calibrated)
    logger.info("Getting predictions...")
    
    # We need to access the raw meta-model predictions to show the "Before" state
    # This requires a bit of a hack since predict_stacking applies calibration automatically if present
    
    # 1. Get Base Model Predictions
    # Ensure features are in the exact order expected by the trained model
    expected_features = architect.catboost_model.feature_names_
    X_holdout = holdout_architect.df[expected_features].copy()
    
    # CatBoost (Handles strings natively if they match training)
    cb_pred = architect.catboost_model.predict_proba(X_holdout)[:, 1]
    
    # LightGBM (Needs 'category' dtype)
    X_lgb = X_holdout.copy()
    for col in architect.cat_features:
        if col in X_lgb.columns:
            X_lgb[col] = X_lgb[col].astype('category')
    lg_pred = architect.lgbm_model.predict(X_lgb)
    
    # XGBoost (Needs encoded integers)
    X_xgb = X_holdout.copy()
    for col in architect.cat_features:
        if col in X_xgb.columns:
            X_xgb[col] = X_xgb[col].astype('category').cat.codes
    xg_pred = architect.xgboost_model.predict_proba(X_xgb)[:, 1]
    
    meta_features = pd.DataFrame({
        'catboost': cb_pred, 
        'lgbm': lg_pred,
        'xgboost': xg_pred
    })
    
    # Uncalibrated (Raw Logistic Regression output)
    y_prob_uncalibrated = architect.meta_model.predict_proba(meta_features)[:, 1]
    
    # Calibrated (Isotonic Regression output)
    y_prob_calibrated = architect.calibrator.transform(y_prob_uncalibrated)
    
    y_true = holdout_architect.df['target']
    
    # --- PLOT 1: Calibration Curve (The "Money Shot") ---
    logger.info("Plotting Calibration Curve...")
    plt.figure(figsize=(10, 6))
    
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, y_prob_uncalibrated, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=10)
    
    plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', linewidth=2, label='Before Calibration')
    plt.plot(prob_pred_cal, prob_true_cal, marker='s', linewidth=2, label='After Calibration (Isotonic)')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (True Fraud Rate)')
    plt.title('Calibration Curve: The Impact of Isotonic Regression')
    plt.legend()
    plt.savefig('plot_calibration.png', dpi=300)
    plt.close()
    
    # --- PLOT 2: ROC Curve ---
    logger.info("Plotting ROC Curve...")
    fpr, tpr, _ = roc_curve(y_true, y_prob_calibrated)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('plot_roc.png', dpi=300)
    plt.close()
    
    # --- PLOT 3: Confusion Matrix ---
    logger.info("Plotting Confusion Matrix...")
    y_pred = (y_prob_calibrated > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 14})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Threshold = 0.5)')
    plt.savefig('plot_confusion_matrix.png', dpi=300)
    plt.close()
    
    # --- PLOT 4: Target Distribution (Refresh) ---
    logger.info("Refreshing Target Distribution...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=features_df)
    plt.title('Class Distribution (0: Legit, 1: Fraud)')
    plt.savefig('plot_target_dist.png', dpi=300)
    plt.close()
    
    # --- PLOT 5: Amount Distribution (Refresh) ---
    logger.info("Refreshing Amount Distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=features_df, x='amount', hue='target', bins=50, log_scale=True, kde=True)
    plt.title('Transaction Amount Distribution by Class (Log Scale)')
    plt.savefig('plot_amount_dist.png', dpi=300)
    plt.close()
    
    logger.info("All plots generated successfully.")

if __name__ == "__main__":
    generate_plots()
