
# 📈 Stock Market Signal Prediction

This project predicts the next-day stock price direction using machine learning ensembles and stacking. It leverages multiple models — **CatBoost**, **XGBoost**, and **LightGBM** — trained on technical indicators from historical stock data.

---

## 🗂 Project Structure

```

stock predictor/
│
├── catboost_info/             # CatBoost metadata
├── saved_models/              # Trained base models
│   ├── all_models.pkl
│   ├── CB1C.cbm
│   ├── XGB1C.json
│   ├── LGBM1C.txt
│   └── LGBM2C.txt
├── stacked_models/            # Stacking models and folds
│   ├── CB1C_fold0.cbm
│   ├── XGB1C_fold0.json
│   └── meta_model.pkl
├── stock predict/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── ensemble model.py          # Train base models & ensemble
├── stacking_model.py          # Train stacking meta-model
├── example of use both.py     # Demo script to load & predict using models
├── submission_ensemble.csv    # Output submission (ensemble)
└── stock2026-public-baseline-v1.ipynb

```

---

## 📊 Dataset Description

### Overview
The dataset is designed to predict the next-day stock price movement using anonymized technical indicators. Each row represents one stock for one trading day.

### Files

| File                     | Rows      | Columns | Description                       |
|---------------------------|-----------|---------|-----------------------------------|
| `train.csv`               | 440,402   | 30      | Training data with features + target |
| `test.csv`                | 53,276    | 29      | Test data without target           |
| `sample_submission.csv`   | 53,276    | 2       | Submission format template         |

### Data Format

**train.csv**
- `id` – Unique identifier  
- `stock_id` – Stock identifier  
- 27 engineered features  
- `target` – Binary label (0 or 1)  

**Example:**
```

id,stock_id,return_1d,return_5d,...,atr_14,target
0,stock_053,-0.03722,-0.00774,...,0.173104,0

```
**Training period:** 2000–2023  

**test.csv**
- Same structure but **without `target`**  
**Example:**
```

id,stock_id,return_1d,return_5d,...,atr_14
440402,stock_030,-0.012364,0.028009,...,0.041448

```
**Test period:** 2024–2026

### Prediction Target
- Predict probability that a stock will **increase in price** next day
- 1.0 → High confidence stock goes up  
- 0.0 → High confidence stock goes down  
- 0.5 → Uncertain  

### Technical Indicators
- SMA – Simple Moving Average  
- EMA – Exponential Moving Average  
- MACD – Moving Average Convergence Divergence  
- RSI – Relative Strength Index  
- ATR – Average True Range  
- ROC – Rate of Change  

### Dataset Considerations
- **Feature Correlation:** Many features are highly correlated  
- **Stock Heterogeneity:** Different stocks behave differently  
- **Temporal Shift:** Market conditions change over time  
- **Class Balance:** Target roughly 50/50

### Submission Format
```

id,target
440402,0.5

````

---

## 🛠 Installation

```bash
pip install pandas numpy scikit-learn xgboost catboost lightgbm tqdm
````

---

## 🏃‍♂️ Usage

### 1. Train Base Models

```bash
python "ensemble model.py"
```

* Trains **CatBoost, XGBoost, LightGBM**
* Saves models to `saved_models/`
* Generates ensemble predictions → `submission_ensemble.csv`

### 2. Train Stacking / Meta-Model

```bash
python "stacking_model.py"
```

* Uses out-of-fold predictions of base models
* Trains Logistic Regression meta-model
* Saves stacking models → `stacked_models/`
* Generates final submission → `submission_stacked.csv`

### 3. Load & Predict with Saved Models

```python
import pickle

with open("saved_models/all_models.pkl", "rb") as f:
    data = pickle.load(f)

models = data["models"]
le = data["le"]
```

---

## ⚙️ Model Details

**Base Models:**

* CatBoostClassifier (`CB1C`)
* XGBoostClassifier (`XGB1C`)
* LightGBMClassifier (`LGBM1C`, `LGBM2C`)

**Stacking Meta-Model:** Logistic Regression

* Trained on out-of-fold predictions

**Evaluation Metric:** ROC-AUC

---

## 🔖 Notes

* Ensure datasets are in `stock predict/`
* Models are saved in native formats for faster loading
* Stacking usually yields better performance than simple averaging

---

## 📜 License

Open-source for educational and research purposes.

```

---

✅ This README includes:
- Project overview  
- Dataset info and table  
- Technical indicator explanations  
- Workflow for training, stacking, and predictions  
- Submission format  
- Styled markdown suitable for GitHub  

---


Do you want me to add that diagram?
```
