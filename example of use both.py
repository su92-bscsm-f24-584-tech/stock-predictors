import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

# -------------------------------
# 1. LOAD DATA & ENCODER
# -------------------------------
def load_and_preprocess(csv_path="test.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['id'])
    
    # Critical: Load the exact LabelEncoder used during training
    le = joblib.load("stacked_models/label_encoder.pkl")
    X['stock_id'] = le.transform(X['stock_id'])
    return df, X

# -------------------------------
# 2. FUNCTION: SIMPLE ENSEMBLE (Script 1 Results)
# -------------------------------
def predict_simple_ensemble(X_input):
    """Averages the single models from the /saved_models directory."""
    print("Inference: Using Simple Ensemble...")
    preds = []

    # Load & Predict CatBoost
    cb = CatBoostClassifier()
    cb.load_model("saved_models/CB1C.cbm")
    preds.append(cb.predict_proba(X_input)[:, 1])

    # Load & Predict XGBoost (Using .values to bypass Pandas bug)
    xgb = XGBClassifier()
    xgb.load_model("saved_models/XGB1C.json")
    preds.append(xgb.predict_proba(X_input.values)[:, 1])

    # Load & Predict LightGBM (Both versions)
    for name in ["LGBM1C", "LGBM2C"]:
        bst = lgb.Booster(model_file=f"saved_models/{name}.txt")
        preds.append(bst.predict(X_input))

    return np.mean(preds, axis=0)

# -------------------------------
# 3. FUNCTION: STACKED ENSEMBLE (Script 2 Results)
# -------------------------------
def predict_stacked_ensemble(X_input, n_folds=5):
    """Processes all folds from /stacked_models and applies the meta-model."""
    print(f"Inference: Using {n_folds}-Fold Stacked Ensemble...")
    base_names = ["CB1C", "XGB1C", "LGBM1C", "LGBM2C"]
    meta_features = pd.DataFrame()

    for name in base_names:
        fold_preds = []
        for f in range(n_folds):
            path = f"stacked_models/{name}_fold{f}"
            if name == "CB1C":
                m = CatBoostClassifier().load_model(f"{path}.cbm")
                fold_preds.append(m.predict_proba(X_input)[:, 1])
            elif name == "XGB1C":
                m = XGBClassifier(); m.load_model(f"{path}.json")
                fold_preds.append(m.predict_proba(X_input.values)[:, 1])
            else:
                m = lgb.Booster(model_file=f"{path}.txt")
                fold_preds.append(m.predict(X_input))
        
        # Average folds for this specific architecture
        meta_features[name] = np.mean(fold_preds, axis=0)

    # Final pass through the meta-model (Logistic Regression)
    meta_model = joblib.load("stacked_models/meta_model.pkl")
    return meta_model.predict_proba(meta_features)[:, 1]

# -------------------------------
# 4. EXECUTION
# -------------------------------
if __name__ == "__main__":
    # Load and prepare data
    raw_test, X_test = load_and_preprocess("test.csv")

    # OPTION 1: Results from Simple Ensemble
    # result_preds = predict_simple_ensemble(X_test)
    
    # OPTION 2: Results from Stacked Ensemble (Usually more accurate)
    result_preds = predict_stacked_ensemble(X_test)

    # Save final results
    submission = pd.DataFrame({'id': raw_test['id'], 'target': result_preds})
    submission.to_csv("final_results.csv", index=False)
    print("\nSuccess! Predictions saved to final_results.csv")
    print(submission.head()) # Show top results
