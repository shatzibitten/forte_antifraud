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
                self.df[col] = self.df[col].astype('category')
            else:
                # If categorical features are missing (e.g. no behavior data), fill with 'unknown'
                self.df[col] = 'unknown'
                self.df[col] = self.df[col].astype('category')
        
        # Add categorical features to the list of features to use
        self.model_features = [f for f in self.features if f in self.df.columns] + \
                              [f for f in self.cat_features if f in self.df.columns] + \
                              [f for f in self.df.columns if 'gnn_emb' in f] + \
                              ['anomaly_score', 'anomaly_score_raw']
        
        # Remove duplicates just in case
        self.model_features = list(set(self.model_features))

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

    def optimize_catboost(self, n_trials=20):
        """Optimizes CatBoost hyperparameters using Optuna."""
        import optuna
        from sklearn.model_selection import cross_val_score
        
        logger.info(f"Optimizing CatBoost hyperparameters ({n_trials} trials)...")
        
        X = self.df[self.model_features]
        y = self.df[self.target]
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 8),  # Limit depth to reduce overfitting
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),  # L2 regularization
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'cat_features': [f for f in self.cat_features if f in X.columns],
                'verbose': 0,
                'random_seed': 42,
                'allow_writing_files': False
            }
            
            model = CatBoostClassifier(**params)
            
            # Use TimeSeriesSplit for CV
            tscv = TimeSeriesSplit(n_splits=3)  # Fewer splits for speed
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                if len(np.unique(y_train)) < 2:
                    continue
                    
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
                pred = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, pred)
                scores.append(auc)
            
            return np.mean(scores) if scores else 0.0
        
        study = optuna.create_study(direction='maximize', study_name='catboost_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best CatBoost params: {study.best_params}")
        logger.info(f"Best CV AUC: {study.best_value:.4f}")
        
        return study.best_params

    def train_final_model(self, use_optuna=False, n_trials=20):
        """Trains the model on the entire dataset."""
        logger.info("Training final CatBoost model...")
        X = self.df[self.model_features]
        y = self.df[self.target]
        
        if use_optuna:
            best_params = self.optimize_catboost(n_trials=n_trials)
            params = {
                **best_params,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'cat_features': [f for f in self.cat_features if f in X.columns],
                'verbose': 100,
                'random_seed': 42,
                'allow_writing_files': False
            }
        else:
            params = {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'cat_features': [f for f in self.cat_features if f in X.columns],
                'verbose': 100,
                'random_seed': 42,
                'allow_writing_files': False
            }
        
        self.catboost_model = CatBoostClassifier(**params)
        self.catboost_model.fit(X, y)
        logger.info("Final CatBoost model trained.")
        return self.catboost_model

    def optimize_lightgbm(self, n_trials=20):
        """Optimizes LightGBM hyperparameters using Optuna."""
        import optuna
        import lightgbm as lgb
        
        logger.info(f"Optimizing LightGBM hyperparameters ({n_trials} trials)...")
        
        X = self.df[self.model_features]
        y = self.df[self.target]
        
        # Prepare LightGBM data
        X_lgb = X.copy()
        for col in self.cat_features:
            if col in X_lgb.columns:
                X_lgb[col] = X_lgb[col].astype('category')
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'objective': 'binary',
                'metric': 'auc',
                'verbose': -1,
                'random_state': 42
            }
            
            model = lgb.LGBMClassifier(**params)
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_lgb):
                X_train, X_val = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                if len(np.unique(y_train)) < 2:
                    continue
                    
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                pred = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, pred)
                scores.append(auc)
            
            return np.mean(scores) if scores else 0.0
        
        study = optuna.create_study(direction='maximize', study_name='lightgbm_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best LightGBM params: {study.best_params}")
        return study.best_params

    def train_lightgbm(self, use_optuna=False, n_trials=20):
        """Trains LightGBM model."""
        logger.info("Training LightGBM model...")
        X = self.df[self.model_features]
        y = self.df[self.target]
        
        X_lgb = X.copy()
        for col in self.cat_features:
            if col in X_lgb.columns:
                X_lgb[col] = X_lgb[col].astype('category')
        
        if use_optuna:
            best_params = self.optimize_lightgbm(n_trials=n_trials)
            params = {
                **best_params,
                'objective': 'binary',
                'metric': 'auc',
                'verbose': -1,
                'random_state': 42
            }
        else:
            params = {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'objective': 'binary',
                'metric': 'auc',
                'verbose': -1,
                'random_state': 42
            }
        
        self.lgbm_model = lgb.LGBMClassifier(**params)
        self.lgbm_model.fit(X_lgb, y)
        logger.info("LightGBM model trained.")
        return self.lgbm_model

    def optimize_xgboost(self, n_trials=20):
        """Optimizes XGBoost hyperparameters using Optuna."""
        import optuna
        import xgboost as xgb
        
        logger.info(f"Optimizing XGBoost hyperparameters ({n_trials} trials)...")
        
        X = self.df[self.model_features]
        y = self.df[self.target]
        
        X_xgb = X.copy()
        for col in self.cat_features:
            if col in X_xgb.columns:
                X_xgb[col] = X_xgb[col].astype('category').cat.codes
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'verbosity': 0,
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_xgb):
                X_train, X_val = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                if len(np.unique(y_train)) < 2:
                    continue
                    
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                pred = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, pred)
                scores.append(auc)
            
            return np.mean(scores) if scores else 0.0
        
        study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best XGBoost params: {study.best_params}")
        return study.best_params

    def train_xgboost(self, use_optuna=False, n_trials=20):
        """Trains XGBoost model."""
        logger.info("Training XGBoost model...")
        import xgboost as xgb
        
        X = self.df[self.model_features]
        y = self.df[self.target]
        
        # XGBoost also handles categories, but we encode them
        X_xgb = X.copy()
        for col in self.cat_features:
            if col in X_xgb.columns:
                X_xgb[col] = X_xgb[col].astype('category').cat.codes
        
        if use_optuna:
            best_params = self.optimize_xgboost(n_trials=n_trials)
            params = {
                **best_params,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'verbosity': 0,
                'random_state': 42
            }
        else:
            params = {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'verbosity': 0,
                'random_state': 42
            }
        
        self.xgboost_model = xgb.XGBClassifier(**params)
        
        self.xgboost_model.fit(X_xgb, y)
        logger.info("XGBoost model trained.")
        return self.xgboost_model

    def train_stacking(self):
        """Trains a stacking ensemble with CatBoost, LightGBM, and XGBoost."""
        logger.info("Training Stacking Ensemble (3 base models)...")
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb
        
        X = self.df[self.model_features]
        y = self.df[self.target]
        
        logger.info("Generating out-of-fold predictions...")
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        catboost_preds = np.zeros(len(self.df))
        lgbm_preds = np.zeros(len(self.df))
        xgb_preds = np.zeros(len(self.df))
        
        # Prepare data for different models
        X_lgb = X.copy()
        for col in self.cat_features:
            if col in X_lgb.columns:
                X_lgb[col] = X_lgb[col].astype('category')
        
        X_xgb = X.copy()
        for col in self.cat_features:
            if col in X_xgb.columns:
                X_xgb[col] = X_xgb[col].astype('category').cat.codes
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            X_train_lgb, X_test_lgb = X_lgb.iloc[train_index], X_lgb.iloc[test_index]
            X_train_xgb, X_test_xgb = X_xgb.iloc[train_index], X_xgb.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            if len(np.unique(y_train)) < 2:
                continue
            
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
            
            # XGBoost
            xg = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, verbosity=0, random_state=42)
            xg.fit(X_train_xgb, y_train)
            xgb_preds[test_index] = xg.predict_proba(X_test_xgb)[:, 1]
            
        # Meta-learner training
        mask = catboost_preds != 0
        
        meta_X = pd.DataFrame({
            'catboost': catboost_preds[mask],
            'lgbm': lgbm_preds[mask],
            'xgboost': xgb_preds[mask]
        })
        meta_y = y[mask]
        
        self.meta_model = LogisticRegression()
        self.meta_model.fit(meta_X, meta_y)
        
        logger.info(f"Stacking Meta-Model Coefficients: {self.meta_model.coef_}")
        return self.meta_model

    def predict_stacking(self, X):
        """Predicts using the stacking ensemble with 3 base models."""
        # Ensure X has same format for each model
        X_lgb = X.copy()
        for col in self.cat_features:
            if col in X_lgb.columns:
                X_lgb[col] = X_lgb[col].astype('category')
        
        X_xgb = X.copy()
        for col in self.cat_features:
            if col in X_xgb.columns:
                X_xgb[col] = X_xgb[col].astype('category').cat.codes
                
        cb_pred = self.catboost_model.predict_proba(X)[:, 1]
        lg_pred = self.lgbm_model.predict_proba(X_lgb)[:, 1]
        xg_pred = self.xgboost_model.predict_proba(X_xgb)[:, 1]
        
        meta_features = pd.DataFrame({
            'catboost': cb_pred, 
            'lgbm': lg_pred,
            'xgboost': xg_pred
        })
        return self.meta_model.predict_proba(meta_features)[:, 1]

    def save_models(self, path='models/'):
        """Saves all trained models to disk."""
        import os
        import joblib
        
        if not os.path.exists(path):
            os.makedirs(path)
            
        logger.info(f"Saving models to {path}...")
        
        if self.catboost_model:
            self.catboost_model.save_model(os.path.join(path, 'catboost_model.cbm'))
            
        if self.lgbm_model:
            self.lgbm_model.booster_.save_model(os.path.join(path, 'lightgbm_model.txt'))
            
        if hasattr(self, 'xgboost_model') and self.xgboost_model:
            self.xgboost_model.save_model(os.path.join(path, 'xgboost_model.json'))

        if hasattr(self, 'meta_model'):
            joblib.dump(self.meta_model, os.path.join(path, 'meta_model.joblib'))
            
        logger.info("Models saved successfully.")

    def load_models(self, path='models/'):
        """Loads models from disk."""
        import os
        import joblib
        import xgboost as xgb
        
        logger.info(f"Loading models from {path}...")
        
        if os.path.exists(os.path.join(path, 'catboost_model.cbm')):
            self.catboost_model = CatBoostClassifier()
            self.catboost_model.load_model(os.path.join(path, 'catboost_model.cbm'))
            
        if os.path.exists(os.path.join(path, 'lightgbm_model.txt')):
            self.lgbm_model = lgb.Booster(model_file=os.path.join(path, 'lightgbm_model.txt'))
            
        if os.path.exists(os.path.join(path, 'xgboost_model.json')):
            self.xgboost_model = xgb.XGBClassifier()
            self.xgboost_model.load_model(os.path.join(path, 'xgboost_model.json'))
            
        if os.path.exists(os.path.join(path, 'meta_model.joblib')):
            self.meta_model = joblib.load(os.path.join(path, 'meta_model.joblib'))
            
        logger.info("Models loaded.")

    def predict_stacking_inference(self, X):
        """Predicts using loaded models (handles LGBM booster difference)."""
        # Filter features to match what the model expects
        if self.catboost_model:
            expected_features = self.catboost_model.feature_names_
            X_filtered = X[expected_features].copy()
        else:
            X_filtered = X.copy()

        # Ensure X has same format
        X_lgb = X_filtered.copy()
        for col in self.cat_features:
            if col in X_lgb.columns:
                X_lgb[col] = X_lgb[col].astype('category')
        
        X_xgb = X_filtered.copy()
        for col in self.cat_features:
            if col in X_xgb.columns:
                X_xgb[col] = X_xgb[col].astype('category').cat.codes
                
        cb_pred = self.catboost_model.predict_proba(X_filtered)[:, 1]
        
        # LGBM Booster predicts raw scores or probs depending on objective. 
        lg_pred = self.lgbm_model.predict(X_lgb)
        
        # XGBoost
        xg_pred = self.xgboost_model.predict_proba(X_xgb)[:, 1]
        
        meta_features = pd.DataFrame({
            'catboost': cb_pred, 
            'lgbm': lg_pred,
            'xgboost': xg_pred
        })
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
