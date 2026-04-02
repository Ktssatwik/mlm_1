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


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


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


def main() -> None:
    model = None
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

    while True:
        print("\n=== Menu ===")
        print("1. Diabetes prediction")
        print("0. Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            diabetes_prediction_flow(model)
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1 or 0.")


if __name__ == "__main__":
    main()
