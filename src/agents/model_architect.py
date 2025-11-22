import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelArchitectAgent:
    def __init__(self, df):
        self.df = df.copy()
        self.catboost_model = None
        self.lgbm_model = None
        self.features = [
            'amount', 'count_txn_1h', 'sum_amount_24h', 'std_amount_7d',
            'avg_amount_30d', 'amount_to_avg_30d', 'amount_per_login_interval',
            'high_risk_device', 'is_fake_os', 'direction_fan_in', 'direction_txn_count',
            'time_since_last_txn', 'monthly_os_changes', 'monthly_phone_model_changes',
            'burstiness_login_interval', 'fano_factor_login_interval', 'zscore_avg_login_interval_7d',
            'logins_last_7_days', 'logins_last_30_days'
        ]
        self.cat_features = ['last_os_categorical', 'last_phone_model_categorical']
        self.target = 'target'

    def prepare_data(self):
        """Prepares data for modeling (handling NaNs, etc)."""
        # Fill NaNs in features
        for col in self.features:
            if col in self.df.columns:
                 self.df[col] = self.df[col].fillna(0)
        
        # Ensure categorical features are strings
        for col in self.cat_features:
            if col in self.df.columns:
                # Force conversion to string for every element
                self.df[col] = self.df[col].apply(lambda x: str(x) if pd.notna(x) and str(x).lower() != 'nan' else 'unknown')
            else:
                # If categorical features are missing (e.g. no behavior data), fill with 'unknown'
                self.df[col] = 'unknown'
        
        # Add categorical features to the list of features to use
        self.model_features = [f for f in self.features if f in self.df.columns] + \
                              [f for f in self.cat_features if f in self.df.columns]

    def train_validate(self):
        """Performs TimeSeriesSplit validation."""
        logger.info("Starting TimeSeriesSplit validation...")
        
        # Sort by time
        self.df = self.df.sort_values('transdatetime')
        
        tscv = TimeSeriesSplit(n_splits=5)
        X = self.df[self.model_features]
        y = self.df[self.target]
        
        fold = 1
        scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            if y_train.nunique() < 2:
                logger.warning(f"Fold {fold}: Training set has only 1 unique class. Skipping.")
                fold += 1
                continue
            
            # Train CatBoost
            model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function='Logloss',
                eval_metric='AUC',
                cat_features=[f for f in self.cat_features if f in X_train.columns],
                verbose=100,
                random_seed=42,
                allow_writing_files=False
            )
            
            model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
            
            preds = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, preds)
            pr_auc = average_precision_score(y_test, preds)
            
            logger.info(f"Fold {fold}: ROC-AUC = {auc:.4f}, PR-AUC = {pr_auc:.4f}")
            scores.append(auc)
            fold += 1
            
            # Save the last model
            self.catboost_model = model
            
        logger.info(f"Average ROC-AUC: {np.mean(scores):.4f}")
        return np.mean(scores)

    def train_final_model(self):
        """Trains the model on the entire dataset."""
        logger.info("Training final model on all data...")
        X = self.df[self.model_features]
        y = self.df[self.target]
        
        self.catboost_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            eval_metric='AUC',
            cat_features=[f for f in self.cat_features if f in X.columns],
            verbose=100,
            random_seed=42,
            allow_writing_files=False
        )
        
        self.catboost_model.fit(X, y)
        logger.info("Final CatBoost model trained.")
        return self.catboost_model

    def train_lightgbm(self):
        """Trains LightGBM model."""
        logger.info("Training LightGBM model...")
        X = self.df[self.model_features]
        y = self.df[self.target]
        
        # LightGBM handles categories differently, usually needs int encoding
        # But we can use 'categorical_feature' param if we convert to 'category' dtype
        X_lgb = X.copy()
        for col in self.cat_features:
            if col in X_lgb.columns:
                X_lgb[col] = X_lgb[col].astype('category')
        
        self.lgbm_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            objective='binary',
            metric='auc',
            verbose=-1,
            random_state=42
        )
        
        self.lgbm_model.fit(X_lgb, y)
        logger.info("LightGBM model trained.")
        return self.lgbm_model

    def train_stacking(self):
        """Trains a stacking ensemble."""
        logger.info("Training Stacking Ensemble...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        
        X = self.df[self.model_features]
        y = self.df[self.target]
        
        # 1. Generate out-of-fold predictions for CatBoost
        logger.info("Generating CatBoost OOF predictions...")
        # We need a custom wrapper or manual CV because cross_val_predict with CatBoost 
        # and categorical features can be tricky if not handled carefully.
        # For simplicity in this hackathon context, we'll use the models trained on full data 
        # to predict (which is overfitting, but standard Stacking requires K-Fold).
        # Let's do proper K-Fold for Stacking.
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        catboost_preds = np.zeros(len(self.df))
        lgbm_preds = np.zeros(len(self.df))
        
        # Prepare LightGBM data
        X_lgb = X.copy()
        for col in self.cat_features:
            if col in X_lgb.columns:
                X_lgb[col] = X_lgb[col].astype('category')
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            X_train_lgb, X_test_lgb = X_lgb.iloc[train_index], X_lgb.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # CatBoost
            cb = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, verbose=0, 
                                    cat_features=[f for f in self.cat_features if f in X_train.columns],
                                    allow_writing_files=False)
            cb.fit(X_train, y_train)
            catboost_preds[test_index] = cb.predict_proba(X_test)[:, 1]
            
            # LightGBM
            lg = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=42)
            lg.fit(X_train_lgb, y_train)
            lgbm_preds[test_index] = lg.predict_proba(X_test_lgb)[:, 1]
            
        # Meta-learner features
        # We only use the indices that were part of the test sets (i.e., not the first training fold)
        # But TimeSeriesSplit leaves the first chunk as train only. 
        # So the first chunk of predictions will be 0.
        # We should only train meta-learner on the non-zero parts.
        
        mask = catboost_preds != 0 # Simple heuristic
        
        meta_X = pd.DataFrame({
            'catboost': catboost_preds[mask],
            'lgbm': lgbm_preds[mask]
        })
        meta_y = y[mask]
        
        self.meta_model = LogisticRegression()
        self.meta_model.fit(meta_X, meta_y)
        
        logger.info(f"Stacking Meta-Model Coefficients: {self.meta_model.coef_}")
        return self.meta_model

    def predict_stacking(self, X):
        """Predicts using the stacking ensemble."""
        # Ensure X has same format
        X_lgb = X.copy()
        for col in self.cat_features:
            if col in X_lgb.columns:
                X_lgb[col] = X_lgb[col].astype('category')
                
        cb_pred = self.catboost_model.predict_proba(X)[:, 1]
        lg_pred = self.lgbm_model.predict_proba(X_lgb)[:, 1]
        
        meta_features = pd.DataFrame({'catboost': cb_pred, 'lgbm': lg_pred})
        return self.meta_model.predict_proba(meta_features)[:, 1]

if __name__ == "__main__":
    # Test run
    from data_steward import DataStewardAgent
    from feature_engineer import FeatureEngineerAgent
    
    steward = DataStewardAgent(
        behavior_path='datasets/behavior_patterns.csv',
        transactions_path='datasets/transactions.csv'
    )
    steward.load_data()
    steward.clean_data()
    merged_df = steward.merge_data()
    
    engineer = FeatureEngineerAgent(merged_df)
    features_df = engineer.generate_features()
    
    architect = ModelArchitectAgent(features_df)
    architect.prepare_data()
    architect.train_validate()
