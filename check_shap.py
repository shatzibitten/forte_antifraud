import logging
import pandas as pd
from catboost import CatBoostClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_importance():
    try:
        # Load model
        model = CatBoostClassifier()
        model.load_model('models/catboost_model.cbm')
        
        # Get feature importance
        importance = model.get_feature_importance()
        feature_names = model.feature_names_
        
        # Create DataFrame
        df_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
        df_imp = df_imp.sort_values('importance', ascending=False)
        
        print("\nTOP 10 FEATURES:")
        print(df_imp.head(10))
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    check_importance()
