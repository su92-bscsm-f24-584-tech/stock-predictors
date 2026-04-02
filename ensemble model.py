import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation

# -------------------------------
# Load dataset
# -------------------------------
train = pd.read_csv(r"stock predict\train.csv")
test = pd.read_csv(r"stock predict\test.csv")

# Encode stock_id
le = LabelEncoder()
all_stocks = pd.concat([train['stock_id'], test['stock_id']])
le.fit(all_stocks)
train['stock_id'] = le.transform(train['stock_id'])
test['stock_id'] = le.transform(test['stock_id'])

# -------------------------------
# Train/Validation split (no shuffle for time-series)
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
# Define models
# -------------------------------
Mdl_master = {
    "CB1C": CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        max_depth=4,
        objective="Logloss",
        eval_metric="AUC",
        verbose=0,
        l2_leaf_reg=0.70,
        colsample_bylevel=0.65,
        thread_count=-1,
        random_state=42
    ),
    "XGB1C" :XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        objective="binary:logistic",
        eval_metric="auc",   # <-- set here
        colsample_bytree=0.55,
        verbosity=0,
        reg_lambda=2.5,
        reg_alpha=0.01,
        random_state=42,
        enable_categorical=True,
        early_stopping_rounds=50,
        use_label_encoder=False
    ),
    "LGBM1C": LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.01,
        max_depth=5,
        objective="binary",
        eval_metric="auc",
        colsample_bytree=0.65,
        reg_lambda=1.75,
        reg_alpha=0.10,
        random_state=42,
        verbosity=-1
    ),
    "LGBM2C": LGBMClassifier(
        n_estimators=800,
        learning_rate=0.0125,
        max_depth=6,
        objective="binary",
        eval_metric="auc",
        subsample=0.8,   # goss approximation replaced with subsample for simplicity
        colsample_bytree=0.70,
        reg_lambda=1.25,
        reg_alpha=0.10,
        random_state=42,
        verbosity=-1
    )
}

# -------------------------------
# Prepare eval sets
# -------------------------------
eval_set = [(X_valid, Y_valid)]      # XGB / LGB
eval_set_cb = (X_valid, Y_valid)     # CatBoost

# -------------------------------
# Train models & predict
# -------------------------------
Preds_Val = {}
Preds_Test = {}

for name, model in Mdl_master.items():
    print(f"\nTraining {name}...")
    
    if name == "CB1C":
        model.fit(X_train, Y_train, eval_set=eval_set_cb, verbose=50)
        preds_val = model.predict_proba(X_valid)[:, 1]
        preds_test = model.predict_proba(X_test)[:, 1]

    elif name == "XGB1C":
        # USE .values FOR BOTH FIT AND PREDICT
        model.fit(
            X_train.values, Y_train.values,
            eval_set=[(X_valid.values, Y_valid.values)],
            verbose=50
        )
        preds_val = model.predict_proba(X_valid.values)[:, 1]
        preds_test = model.predict_proba(X_test.values)[:, 1]
                
    else:  # LightGBM
        model.fit(
            X_train, Y_train,
            eval_set=eval_set,
            callbacks=[early_stopping(50), log_evaluation(50)]
        )
        preds_val = model.predict_proba(X_valid)[:, 1]
        preds_test = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(Y_valid, preds_val)
    print(f"{name} Validation AUC: {auc:.4f}")
    
    Preds_Val[name] = preds_val
    Preds_Test[name] = preds_test

    
    auc = roc_auc_score(Y_valid, preds_val)
    print(f"{name} Validation AUC: {auc:.4f}")
    
    Preds_Val[name] = preds_val
    Preds_Test[name] = preds_test

# -------------------------------
# Ensemble predictions (average)
# -------------------------------
ensemble_val = np.mean(list(Preds_Val.values()), axis=0)
ensemble_test = np.mean(list(Preds_Test.values()), axis=0)

print("Ensemble Validation AUC:", roc_auc_score(Y_valid, ensemble_val))

# -------------------------------
# Save submission
# -------------------------------
submission = pd.DataFrame({
    'id': test['id'],
    'target': ensemble_test
})
submission.to_csv("submission_ensemble.csv", index=False)
print("Submission file created: submission_ensemble.csv")
import pickle
import os

# Create a directory for models
os.makedirs("saved_models", exist_ok=True)

for name, model in Mdl_master.items():
    print(f"Saving {name}...")
    
    if name == "CB1C":
        # CatBoost native format
        model.save_model(f"saved_models/{name}.cbm")
        
    elif name == "XGB1C":
        # XGBoost native format (JSON or UBJ)
        model.save_model(f"saved_models/{name}.json")
        
    else:  # LightGBM
        # LightGBM native format       
        model.booster_.save_model(f"saved_models/{name}.txt")

# Save the meta-info and encoder
with open("saved_models/all_models.pkl", "wb") as f:
    pickle.dump({"models": Mdl_master, "le": le}, f)

print("All models saved successfully!")
