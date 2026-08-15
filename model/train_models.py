"""Train and evaluate all required classification models for ML Assignment 2."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "model"
DATA_DIR = PROJECT_DIR
DATA_URL = "https://archive.ics.uci.edu/static/public/17/data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    "radius1", "texture1", "perimeter1", "area1", "smoothness1",
    "compactness1", "concavity1", "concave_points1", "symmetry1",
    "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2",
    "smoothness2", "compactness2", "concavity2", "concave_points2",
    "symmetry2", "fractal_dimension2", "radius3", "texture3", "perimeter3",
    "area3", "smoothness3", "compactness3", "concavity3",
    "concave_points3", "symmetry3", "fractal_dimension3"
]
ALL_COLUMNS = ["ID", "Diagnosis"] + FEATURE_COLUMNS

MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
}


def load_dataset() -> pd.DataFrame:
    """Download the UCI dataset. Handles both header/no-header versions."""
    df = pd.read_csv(DATA_URL)
    if "Diagnosis" not in df.columns:
        df = pd.read_csv(DATA_URL, header=None, names=ALL_COLUMNS)
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df[ALL_COLUMNS].copy()


def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_dataset()

    # Target encoding: benign = 0, malignant = 1.
    df["Diagnosis"] = df["Diagnosis"].map({"B": 0, "M": 1})
    if df["Diagnosis"].isna().any():
        raise ValueError("Unexpected diagnosis labels found in the UCI dataset.")

    X = df[FEATURE_COLUMNS].copy()
    y = df["Diagnosis"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_data = X_test.copy()
    # Preserve ID for traceability in the submitted test_data.csv.
    test_data.insert(0, "ID", df.loc[X_test.index, "ID"].values)
    test_data["Diagnosis"] = y_test.map({0: "B", 1: "M"}).values
    test_data.to_csv(DATA_DIR / "test_data.csv", index=False)

    metrics_rows = []
    metadata = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "dataset_url": DATA_URL,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_instances": int(len(X_train)),
        "test_instances": int(len(X_test)),
        "feature_count": len(FEATURE_COLUMNS),
        "positive_class": "M (malignant)",
        "negative_class": "B (benign)",
        "models": list(MODELS.keys()),
    }

    for model_name, model in MODELS.items():
        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metric_values = calculate_metrics(y_test, y_pred, y_prob)
        metrics_rows.append({"ML Model Name": model_name, **metric_values})
        joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(PROJECT_DIR / "model_metrics.csv", index=False)
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nModel evaluation results:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved test data to: {DATA_DIR / 'test_data.csv'}")
    print(f"Saved model files to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
