# 🏠 Real Estate Investment Advisor
### Predicting Property Profitability & Future Value

**🚀 Live app: [shriyam-realestate-advisor.streamlit.app](https://shriyam-realestate-advisor.streamlit.app)**

A machine-learning app that helps investors decide:
1. **Is this property a Good Investment?** — classification
2. **What will it be worth in 5 years?** — regression

Built with Python, Pandas, Scikit-learn, XGBoost, MLflow and Streamlit, on 250,000 Indian property listings (20 states, 42 cities, 500 localities).

---

## 📁 Project structure
```
├── app.py                     # Streamlit app (4 pages)
├── src/
│   ├── features.py            # shared feature engineering + target rules (used by training AND app)
│   ├── preprocess.py          # Step 1 – cleaning, encoding, feature engineering, targets
│   ├── eda.py                 # Step 2 – 20 EDA questions → reports/eda/*.png + eda_summary.md
│   └── train.py               # Step 3 – 6 models, MLflow tracking + model registry
├── data/
│   ├── india_housing_prices.csv   # raw dataset (add yourself, not in git)
│   ├── processed_housing.csv      # cleaned + engineered dataset (deliverable)
│   └── app_data.parquet           # light copy the app loads
├── models/                    # best models (.joblib), city medians, metrics.json
├── reports/                   # EDA charts, confusion matrix, actual-vs-predicted, app & MLflow screenshots
├── .streamlit/config.toml     # theme
└── requirements.txt
```

## ▶️ How to run
```bash
pip install -r requirements.txt
python src/preprocess.py      # ~20 s
python src/eda.py             # ~30 s
python src/train.py           # ~5 min — creates mlflow.db + mlruns/
streamlit run app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse experiments at http://localhost:5000
```
`models/` already contains trained models, so `streamlit run app.py` works straight away.

---

## 1. Data preprocessing
| Step | What was done |
|---|---|
| Missing values | None in the data; pipeline still imputes median (numeric) / mode (categorical) |
| Duplicates | Checked on all columns except ID → 0 found |
| Data-quality fix | 116,304 rows had `Floor_No > Total_Floors` → capped at Total_Floors |
| Unit fix | Raw `Price_per_SqFt` was in lakhs/sqft rounded to 2 dp → recomputed in ₹/sqft |
| Outliers | IQR on ₹/sqft → 19,723 rows (7.9 %) flagged in `PPSF_Outlier`, kept (so any user input can be scored) |
| Encoding | One-hot for City, Property_Type, Furnished, Facing, Owner_Type, Transport, Availability; binary flags for Parking/Security/Ready |
| Scaling | `StandardScaler` on all numeric features, inside the sklearn Pipeline |

**Engineered features:** Price_per_SqFt (₹), Amenity_Count + one flag per amenity, Transport_Score (1–3), School_Density_Score, Hospital_Density_Score, **Infrastructure_Score** (0.35·schools + 0.35·hospitals + 0.30·transport), Floor_Ratio, SqFt_per_BHK, City_Median_PPSF, **PPSF_vs_City** (how cheap/expensive vs its city).

## 2. Target definitions
The dataset has no historical sale prices, so both targets are built from domain rules (the brief's suggested approach).

**Future price (regression)** — feature-based growth rate:
`Future_Price_5Y = Price × (1 + r)^5`, where `r` starts at **8 %** and adjusts:
transport ±1 %, infrastructure ±1 %, under-construction +1 %, age −0.06 %/yr over 15 yrs, amenities ±0.3 % each, security +0.5 %, under-priced vs city median up to +1.5 %, villa +0.5 % / house +0.3 %. Clipped to 3–14 %.

**Good Investment (classification)** — multi-factor score, 7 checks, **≥ 4 = Good**:
price/sqft ≤ city median · BHK ≥ 3 · Ready to move · transport Medium/High · 3+ amenities · security · growth rate ≥ 8 %.
Result: 65 % Good / 35 % Not Good (stratified split used).

## 3. EDA — key findings
Full answers to all 20 questions: [`reports/eda_summary.md`](reports/eda_summary.md), charts in `reports/eda/`.
- Prices (₹10 L–₹5 Cr) and sizes (500–5,000 sqft) are almost **uniformly distributed**; size vs price r ≈ 0.
- Location, property type, facing, furnishing and parking barely move ₹/sqft (state spread ≈ 2 %).
- Raw features are nearly **independent** (|r| < 0.05) → the dataset is synthetic; current price carries little signal from features.
- What *does* separate good from poor investments: **public transport** (Good share Low 37 % → High 83 %), **amenities** (45 % → 80 %), **security** (51 % → 79 %), and **infrastructure** (corr. with growth rate 0.52).
- No crime-rate column exists, so the **Security** flag is used as the safety proxy.

## 4. Models & results (20 % hold-out test set, 50,000 rows)
**Classification — Good Investment**
| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.905 | 0.928 | 0.922 | 0.934 | 0.970 |
| Random Forest | 0.978 | 0.984 | 0.979 | 0.988 | 0.998 |
| **XGBoost** ✅ | **0.995** | **0.996** | **0.995** | **0.997** | **1.000** |

**Regression — Price after 5 years (₹ Lakhs)**
| Model | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | 15.72 | 11.71 | 0.9942 | 7.67 % |
| Random Forest | 7.22 | 5.23 | 0.9988 | 1.46 % |
| **XGBoost** ✅ | **2.10** | **1.62** | **0.9999** | **0.62 %** |

> **Why are scores so high?** Both targets are deterministic functions of the input features, so a strong model can learn the rules almost perfectly. The useful comparison is *between* models: the linear models can't capture the thresholds and the compounding interaction, and tree boosting can. On real transaction data, expect much lower numbers.

Top drivers (XGBoost classifier): transport access, security, amenity count, clubhouse, BHK, price vs city median.

## 5. MLflow
- Two experiments: `RealEstate_GoodInvestment_Classification`, `RealEstate_FuturePrice_Regression`
- Each run logs params, all metrics, a confusion-matrix / actual-vs-predicted plot and the full pipeline
- Best models registered as **`GoodInvestmentClassifier`** and **`FuturePriceRegressor`**
- Screenshot: `reports/mlflow_clf_runs.png`, `reports/mlflow_reg_runs.png`

## 6. Streamlit app
| Page | What it does |
|---|---|
| **Predict a property** | Form → Good/Not Good verdict, confidence %, probability gauge, 5-yr price, implied annual growth vs 8 % baseline, 7-point "why" checklist, similar listings |
| **Explore listings** | Filter by state, city, type, price, size, BHK, good-only; KPIs; sortable table; CSV download |
| **Market insights** | State→city treemap heatmap, city×BHK heatmap, trend by build year, transport & infrastructure charts |
| **Model performance** | Metrics for all 6 models, confusion matrix, actual-vs-predicted, feature importance |

## 7. Business use cases
- **Investors** — screen a listing in seconds for long-term return.
- **Buyers** — find under-priced, well-connected properties in developing areas.
- **Real-estate platforms** — auto-score every listing with a "Good Investment" badge and 5-yr forecast.
- **Trust** — the checklist explains each verdict, so recommendations are transparent.

## 8. Limitations & next steps
- Targets are rule-based; with real resale/transaction history the regression target should be replaced by actual appreciation.
- Locality is not used as a feature (500 synthetic IDs with no signal); real geo-coordinates would enable proper map heatmaps.
- Possible additions: SHAP explanations per prediction, hyperparameter search logged to MLflow, Streamlit Cloud deployment.
