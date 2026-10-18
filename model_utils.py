"""Training helpers shared by train.py and app.py."""
import warnings
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEATURES = ["Temperature", "Humidity", "Pressure", "WindSpeed", "CloudCover", "RainToday"]
TARGET = "RainTomorrow"


def _pipeline():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def train_bundle(csv_path="rain.csv"):
    """Evaluate on a held-out split, then refit on all data for deployment."""
    df = pd.read_csv(csv_path).drop_duplicates()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = _pipeline()
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
    model.fit(X_tr, y_tr)
    pred, proba = model.predict(X_te), model.predict_proba(X_te)[:, 1]
    metrics = dict(
        accuracy=accuracy_score(y_te, pred), roc_auc=roc_auc_score(y_te, proba),
        precision=precision_score(y_te, pred), recall=recall_score(y_te, pred),
        f1=f1_score(y_te, pred), cv_accuracy=float(cv_acc.mean()),
        cv_std=float(cv_acc.std()), confusion=confusion_matrix(y_te, pred).tolist(),
        n_train=len(X_tr), n_test=len(X_te))

    model.fit(X, y)  # final model sees all rows
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    return dict(
        model=model, features=FEATURES, metrics=metrics,
        coefs=dict(zip(FEATURES, lr.coef_[0].tolist())),
        means=dict(zip(FEATURES, scaler.mean_.tolist())),
        stds=dict(zip(FEATURES, scaler.scale_.tolist())),
        ranges={c: [float(df[c].min()), float(df[c].max())] for c in FEATURES})


def save_bundle(path="rain_model.pkl", csv_path="rain.csv"):
    bundle = train_bundle(csv_path)
    joblib.dump(bundle, path)
    return bundle
