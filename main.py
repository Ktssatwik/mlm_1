import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


DATA_PATH = Path("dataset") / "diabetes.csv"
MODEL_PATH = Path("mlm(s)") / "diabetes_model.joblib"
MODEL_PATH_LEGACY = Path("mlm(s)") / "diabeties_model.joblib"
HEART_DATA_PATH = Path("dataset") / "heart.csv"
HEART_MODEL_PATH = Path("mlm(s)") / "heart_model.joblib"

# We are intentionally excluding Insulin as requested.
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# In these columns, 0 is treated as invalid and converted to NaN.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "BMI"]

HEART_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def load_heart_data() -> pd.DataFrame:
    if not HEART_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {HEART_DATA_PATH}")
    return pd.read_csv(HEART_DATA_PATH)


def build_model(df: pd.DataFrame):
    work_df = df.copy()
    x = work_df[FEATURES].copy()
    y = work_df["Outcome"].copy()

    for col in ZERO_AS_MISSING:
        x[col] = x[col].replace(0, np.nan)

    if HAS_IMBLEARN:
        model = ImbPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=69)),
                # Aligned to tuned LR settings from diabetes.ipynb best params.
                ("lr", LogisticRegression(C=2, solver="lbfgs", max_iter=1000, random_state=69)),
            ]
        )
    else:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=2, solver="lbfgs", max_iter=1000, random_state=69)),
            ]
        )

    model.fit(x, y)
    return model


def build_heart_model(df: pd.DataFrame):
    work_df = df.copy()

    # Remove duplicate rows to reduce train/eval contamination risk.
    work_df = work_df.drop_duplicates()

    x = work_df[HEART_FEATURES].copy()
    y = work_df["target"].copy()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            # Aligned to tuned LR settings from heart.ipynb best params.
            ("lr", LogisticRegression(C=0.9, solver="lbfgs", max_iter=1000, random_state=69)),
        ]
    )

    model.fit(x, y)
    return model


def ask_float(label: str, min_value: float | None = None) -> float:
    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = float(raw)
            if min_value is not None and value < min_value:
                print(f"Please enter a value >= {min_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


def get_user_features() -> pd.DataFrame:
    print("\nEnter patient values:")
    row = {
        "Pregnancies": ask_float("Pregnancies", min_value=0),
        "Glucose": ask_float("Glucose", min_value=0),
        "BloodPressure": ask_float("BloodPressure", min_value=0),
        "SkinThickness": ask_float("SkinThickness", min_value=0),
        "BMI": ask_float("BMI", min_value=0),
        "DiabetesPedigreeFunction": ask_float("DiabetesPedigreeFunction", min_value=0),
        "Age": ask_float("Age", min_value=0),
    }
    return pd.DataFrame([row], columns=FEATURES)


def get_heart_user_features() -> pd.DataFrame:
    print("\nEnter patient values:")
    row = {
        "age": ask_float("age", min_value=0),
        "sex": ask_float("sex (0=female, 1=male)", min_value=0),
        "cp": ask_float("cp", min_value=0),
        "trestbps": ask_float("trestbps", min_value=0),
        "chol": ask_float("chol", min_value=0),
        "fbs": ask_float("fbs (0/1)", min_value=0),
        "restecg": ask_float("restecg", min_value=0),
        "thalach": ask_float("thalach", min_value=0),
        "exang": ask_float("exang (0/1)", min_value=0),
        "oldpeak": ask_float("oldpeak", min_value=0),
        "slope": ask_float("slope", min_value=0),
        "ca": ask_float("ca", min_value=0),
        "thal": ask_float("thal", min_value=0),
    }
    return pd.DataFrame([row], columns=HEART_FEATURES)


def diabetes_prediction_flow(model) -> None:
    user_x = get_user_features()
    prob = float(model.predict_proba(user_x)[0][1])
    pred = int(model.predict(user_x)[0])

    print("\nPrediction result:")
    if pred == 1:
        print("Diabetes: YES")
    else:
        print("Diabetes: NO")
    print(f"Probability of diabetes: {prob:.4f}")


def heart_prediction_flow(model) -> None:
    user_x = get_heart_user_features()
    prob = float(model.predict_proba(user_x)[0][1])
    pred = int(model.predict(user_x)[0])

    print("\nPrediction result:")
    if pred == 1:
        print("Heart disease: YES")
    else:
        print("Heart disease: NO")
    print(f"Probability of heart disease: {prob:.4f}")


def main() -> None:
    model = None
    heart_model = None
    load_path = MODEL_PATH if MODEL_PATH.exists() else MODEL_PATH_LEGACY
    if load_path.exists():
        try:
            model = joblib.load(load_path)
        except Exception:
            model = None

    # If model artifact is not available, build using the same LR setup used in diabetes.ipynb.
    if model is None:
        try:
            df = load_data()
            model = build_model(df)
            try:
                joblib.dump(model, MODEL_PATH)
            except Exception:
                pass
        except Exception as exc:
            print(f"Failed to initialize diabetes model: {exc}")
            sys.exit(1)

    if HEART_MODEL_PATH.exists():
        try:
            heart_model = joblib.load(HEART_MODEL_PATH)
        except Exception:
            heart_model = None

    if heart_model is None:
        try:
            heart_df = load_heart_data()
            heart_model = build_heart_model(heart_df)
            try:
                joblib.dump(heart_model, HEART_MODEL_PATH)
            except Exception:
                pass
        except Exception as exc:
            print(f"Failed to initialize heart model: {exc}")
            sys.exit(1)

    while True:
        print("\n=== Menu ===")
        print("1. Diabetes prediction")
        print("2. Heart disease prediction")
        print("0. Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            diabetes_prediction_flow(model)
        elif choice == "2":
            heart_prediction_flow(heart_model)
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 0.")


if __name__ == "__main__":
    main()
