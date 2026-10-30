"""
Step 3 — Model training, evaluation & MLflow tracking

Classification : Good_Investment   (LogReg, RandomForest, XGBoost)
Regression     : Future_Price_5Y   (LinearRegression, RandomForest, XGBoost)

Every run logs params, metrics (Accuracy/F1/ROC-AUC/Precision/Recall | RMSE/MAE/R²),
a confusion matrix / residual plot, and the fitted pipeline.
The best model of each task is registered in the MLflow Model Registry and saved to
models/ for the Streamlit app.

Run:  python src/train.py
View: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from features import CAT_FEATURES, MODEL_FEATURES, NUM_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed_housing.csv"
MODELS = ROOT / "models"
PLOTS = ROOT / "reports" / "models"
PLOTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
SEED = 42


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


def feature_names(pipe):
    return list(pipe.named_steps["prep"].get_feature_names_out())


def importances(pipe):
    m = pipe.named_steps["model"]
    if hasattr(m, "feature_importances_"):
        vals = m.feature_importances_
    else:
        vals = np.abs(np.ravel(m.coef_))
    s = pd.Series(vals, index=feature_names(pipe))
    s.index = s.index.str.replace(r"^(num|cat)__", "", regex=True)
    return (s / s.sum()).sort_values(ascending=False)


# ------------------------------------------------------------------ classification
def run_classification(X_tr, X_te, y_tr, y_te):
    mlflow.set_experiment("RealEstate_GoodInvestment_Classification")
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=2000, C=1.0), {"C": 1.0}),
        "RandomForest": (RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=5,
                                                n_jobs=-1, random_state=SEED),
                         {"n_estimators": 150, "max_depth": 18, "min_samples_leaf": 5}),
        "XGBoost": (XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
                                  colsample_bytree=0.8, n_jobs=-1, random_state=SEED, eval_metric="logloss"),
                    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1}),
    }
    results = {}
    for name, (est, params) in models.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline([("prep", preprocessor()), ("model", est)]).fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            proba = pipe.predict_proba(X_te)[:, 1]
            m = {"accuracy": accuracy_score(y_te, pred), "f1": f1_score(y_te, pred),
                 "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
                 "roc_auc": roc_auc_score(y_te, proba)}
            mlflow.log_params({"model": name, **params, "train_rows": len(X_tr)})
            mlflow.log_metrics(m)

            fig, ax = plt.subplots(figsize=(4.5, 4))
            ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Not Good", "Good"],
                                                    cmap="Greens", ax=ax, colorbar=False)
            ax.set_title(f"{name} — confusion matrix")
            fig.tight_layout(); p = PLOTS / f"cm_{name}.png"; fig.savefig(p, dpi=110); plt.close(fig)
            mlflow.log_artifact(str(p))
            mlflow.sklearn.log_model(pipe, name="model", input_example=X_te.head(3),
                                     serialization_format="cloudpickle")
            results[name] = {"metrics": m, "pipe": pipe, "run_id": run.info.run_id}
            print(f"[CLF] {name:18s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    return results


# ------------------------------------------------------------------ regression
def run_regression(X_tr, X_te, y_tr, y_te):
    mlflow.set_experiment("RealEstate_FuturePrice_Regression")
    models = {
        "LinearRegression": (LinearRegression(), {}),
        "RandomForest": (RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=5,
                                               n_jobs=-1, random_state=SEED),
                         {"n_estimators": 150, "max_depth": 18, "min_samples_leaf": 5}),
        "XGBoost": (XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.08, subsample=0.9,
                                 colsample_bytree=0.8, n_jobs=-1, random_state=SEED),
                    {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.08}),
    }
    results = {}
    for name, (est, params) in models.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline([("prep", preprocessor()), ("model", est)]).fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            m = {"rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
                 "mae": mean_absolute_error(y_te, pred), "r2": r2_score(y_te, pred),
                 "mape_pct": float(np.mean(np.abs((y_te - pred) / y_te)) * 100)}
            mlflow.log_params({"model": name, **params, "train_rows": len(X_tr)})
            mlflow.log_metrics(m)

            fig, ax = plt.subplots(figsize=(5, 4))
            idx = np.random.RandomState(0).choice(len(pred), 4000, replace=False)
            ax.scatter(np.asarray(y_te)[idx], pred[idx], s=4, alpha=.4)
            lim = [0, max(y_te.max(), pred.max())]
            ax.plot(lim, lim, "r--", lw=1)
            ax.set_xlabel("Actual future price (₹ L)"); ax.set_ylabel("Predicted (₹ L)")
            ax.set_title(f"{name} — actual vs predicted")
            fig.tight_layout(); p = PLOTS / f"avp_{name}.png"; fig.savefig(p, dpi=110); plt.close(fig)
            mlflow.log_artifact(str(p))
            mlflow.sklearn.log_model(pipe, name="model", input_example=X_te.head(3),
                                     serialization_format="cloudpickle")
            results[name] = {"metrics": m, "pipe": pipe, "run_id": run.info.run_id}
            print(f"[REG] {name:18s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    return results


def register_best(results, key, higher_better, reg_name, file_name):
    best = (max if higher_better else min)(results, key=lambda n: results[n]["metrics"][key])
    info = results[best]
    mv = mlflow.register_model(f"runs:/{info['run_id']}/model", reg_name)
    joblib.dump(info["pipe"], MODELS / file_name, compress=3)
    print(f"Best {reg_name}: {best} ({key}={info['metrics'][key]:.4f}) -> registry v{mv.version}")
    return best, importances(info["pipe"])


def main():
    df = pd.read_csv(DATA)
    X = df[MODEL_FEATURES]

    # Classification
    y = df["Good_Investment"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    clf_res = run_classification(X_tr, X_te, y_tr, y_te)
    best_clf, clf_imp = register_best(clf_res, "f1", True, "GoodInvestmentClassifier", "classifier.joblib")

    # Regression
    y = df["Future_Price_5Y"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    reg_res = run_regression(X_tr, X_te, y_tr, y_te)
    best_reg, reg_imp = register_best(reg_res, "rmse", False, "FuturePriceRegressor", "regressor.joblib")

    summary = {
        "classification": {n: r["metrics"] for n, r in clf_res.items()},
        "regression": {n: r["metrics"] for n, r in reg_res.items()},
        "best_classifier": best_clf, "best_regressor": best_reg,
        "clf_importance": clf_imp.head(20).round(4).to_dict(),
        "reg_importance": reg_imp.head(20).round(4).to_dict(),
    }
    (MODELS / "metrics.json").write_text(json.dumps(summary, indent=1))
    print("Saved models + metrics.json")


if __name__ == "__main__":
    main()
