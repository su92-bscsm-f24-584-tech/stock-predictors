import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from tqdm import tqdm
from xgboost.callback import EarlyStopping
import os
import joblib
os.makedirs("stacked_models", exist_ok=True)
# -------------------------------
# Load dataset
# -------------------------------
train = pd.read_csv(r"train.csv")
test = pd.read_csv(r"test.csv")

# Encode stock_id
le = LabelEncoder()
all_stocks = pd.concat([train['stock_id'], test['stock_id']])
le.fit(all_stocks)
train['stock_id'] = le.transform(train['stock_id'])
test['stock_id'] = le.transform(test['stock_id'])

# -------------------------------
# Train/Validation split (time-based)
# -------------------------------
split = int(len(train) * 0.8)
train_df = train.iloc[:split]
valid_df = train.iloc[split:]

X_train = train_df.drop(columns=['target', 'id'])
Y_train = train_df['target']
X_valid = valid_df.drop(columns=['target', 'id'])
Y_valid = valid_df['target']
X_test = test.drop(columns=['id'])

print("Train shape:", X_train.shape, "Validation shape:", X_valid.shape)
print("Train target mean:", Y_train.mean(), "Validation target mean:", Y_valid.mean())

# -------------------------------
# Define base models
# -------------------------------
base_models = {
    "CB1C": CatBoostClassifier(iterations=900, learning_rate=0.01, max_depth=4,
                               objective="Logloss", eval_metric="AUC",
                               verbose=0, l2_leaf_reg=0.7, colsample_bylevel=0.65, random_state=42),
    "XGB1C": XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=5,
                           objective="binary:logistic", eval_metric="auc",
                           colsample_bytree=0.55, verbosity=0, reg_lambda=2.5,
                           reg_alpha=0.01, random_state=42, use_label_encoder=False),
    "LGBM1C": LGBMClassifier(n_estimators=1200, learning_rate=0.01, max_depth=5,
                             objective="binary", eval_metric="auc",
                             colsample_bytree=0.65, reg_lambda=1.75, reg_alpha=0.1, random_state=42),
    "LGBM2C": LGBMClassifier(n_estimators=800, learning_rate=0.0125, max_depth=6,
                             objective="binary", eval_metric="auc",
                             subsample=0.8, colsample_bytree=0.7,
                             reg_lambda=1.25, reg_alpha=0.1, random_state=42)
}

# -------------------------------
# Prepare stacking structures
# -------------------------------
OOF_Preds = pd.DataFrame(index=X_train.index)
Test_Preds = pd.DataFrame(index=X_test.index)

# -------------------------------
# Stratified K-Fold stacking
# -------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# ... (Keep your imports and data loading the same) ...

for name, model in tqdm(base_models.items(), desc="Training base models"):
    oof = np.zeros(X_train.shape[0])
    test_fold_preds = np.zeros((X_test.shape[0], cv.n_splits))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, Y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = Y_train.iloc[tr_idx], Y_train.iloc[val_idx]
        fold_filename = f"stacked_models/{name}_fold{fold}"

        if name == "CB1C":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            test_fold_preds[:, fold] = model.predict_proba(X_test)[:, 1]
            model.save_model(f"{fold_filename}.cbm")
    
        elif name == "XGB1C":
            # 1. ADD early_stopping_rounds TO CONSTRUCTOR
            model.set_params(early_stopping_rounds=50)
            # 2. USE .values TO BYPASS PD.UTIL ERROR
            model.fit(
                X_tr.values, y_tr.values,
                eval_set=[(X_val.values, y_val.values)],
                verbose=False
            )
            oof[val_idx] = model.predict_proba(X_val.values)[:, 1]
            test_fold_preds[:, fold] = model.predict_proba(X_test.values)[:, 1]
            model.save_model(f"{fold_filename}.json")
        else: # LightGBM (LGBM1C, LGBM2C)
            # Use callback for newer LightGBM versions to be safe
            from lightgbm import early_stopping, log_evaluation
            model.fit(
                X_tr, y_tr, 
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[
                    early_stopping(stopping_rounds=50),
                    log_evaluation(period=0) # Set to 0 or False to suppress output
                ]
            )
            
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            test_fold_preds[:, fold] = model.predict_proba(X_test)[:, 1]
            model.booster_.save_model(f"{fold_filename}.txt")
    OOF_Preds[name] = oof
    Test_Preds[name] = test_fold_preds.mean(axis=1)
    print(f"{name} OOF AUC: {roc_auc_score(Y_train, oof):.4f}")

# -------------------------------
# Meta-model & Final Stats
# -------------------------------
# Ridge for binary targets works, but LogisticRegression is often better for stacking probabilities
# ... (Base model training loop from your previous block goes here) ...

# -------------------------------
# Meta-model & Final Stats
# -------------------------------
print("\nTraining Meta-Model (Logistic Regression)...")
from sklearn.linear_model import LogisticRegression

meta_model = LogisticRegression()
meta_model.fit(OOF_Preds, Y_train)

joblib.dump(meta_model, "stacked_models/meta_model.pkl")
joblib.dump(le, "stacked_models/label_encoder.pkl") 
# Final ensemble coefficients 
# coef_[0] is used because LogisticRegression stores coefficients in a 2D array
print("\nMeta-Model Weights:")
for n, coef in zip(base_models.keys(), meta_model.coef_[0]):
    print(f"  {n}: {coef:.4f}")

# Predict on test set using the meta-model
stacked_test_preds = meta_model.predict_proba(Test_Preds)[:, 1]

# -------------------------------
# Save submission
# -------------------------------
submission = pd.DataFrame({
    'id': test['id'],
    'target': stacked_test_preds
})

submission.to_csv("submission_stacked.csv", index=False)
print("\nStacked submission saved: submission_stacked.csv")

# -------------------------------
# Optional: Validation of the Stack
# -------------------------------
# Note: Since the meta-model was trained on OOF_Preds, this AUC 
# is a good estimate of your cross-validation performance.
final_oof_auc = roc_auc_score(Y_train, meta_model.predict_proba(OOF_Preds)[:, 1])
print(f"Final Stacked OOF AUC: {final_oof_auc:.4f}")