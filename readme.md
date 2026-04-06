# Healthcare Risk Prediction (CLI)

A simple command-line machine learning app that predicts:

1. Diabetes risk
2. Heart disease risk

The app is implemented in `main.py` and supports interactive input through a menu.

## Project Structure

- `main.py` - Main CLI app with both prediction flows
- `dataset/diabetes.csv` - Diabetes dataset
- `dataset/heart.csv` - Heart dataset
- `mlm(s)/diabetes_model.joblib` - Saved diabetes model (generated/used by app)
- `mlm(s)/heart_model.joblib` - Saved heart model (generated/used by app)
- `requirements.txt` - Python dependencies
- `mlm(s)/diabetes.ipynb` - Diabetes notebook experiments
- `mlm(s)/heart.ipynb` - Heart notebook experiments

## Features

- Menu-driven CLI interface
- Two models in one app:
  - Model 1: Diabetes prediction
  - Model 2: Heart disease prediction
- Auto-load existing model artifacts if available
- Auto-train and save models if artifacts are missing
- Probability output (`predict_proba`) along with YES/NO prediction

## Model Details

### Model 1: Diabetes

- Algorithm: Logistic Regression
- Pipeline:
  - Median imputation for selected columns
  - StandardScaler
  - SMOTE (if `imblearn` is installed)
  - LogisticRegression (`C=2`, `solver=lbfgs`, `max_iter=1000`, `random_state=69`)
- Input features:
  - `Pregnancies`
  - `Glucose`
  - `BloodPressure`
  - `SkinThickness`
  - `BMI`
  - `DiabetesPedigreeFunction`
  - `Age`

Note: `Insulin` is intentionally excluded. In selected columns, `0` is treated as missing and imputed.

### Model 2: Heart Disease

- Algorithm: Logistic Regression
- Pipeline:
  - Duplicate row removal before training
  - StandardScaler
  - LogisticRegression (`C=0.9`, `solver=lbfgs`, `max_iter=1000`, `random_state=69`)
- Input features:
  - `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

You will see:

- `1` -> Diabetes prediction
- `2` -> Heart disease prediction
- `0` -> Exit

## Example Flow

1. Run `python main.py`
2. Choose a model from the menu
3. Enter patient values when prompted
4. View prediction result and probability

## Notes

- The app expects datasets at:
  - `dataset/diabetes.csv`
  - `dataset/heart.csv`
- If model files are missing, they are trained and saved automatically.
- Legacy diabetes model filename `diabeties_model.joblib` is also supported for loading.

## Future Improvements

- Add input validation for categorical heart fields (`cp`, `restecg`, `slope`, `thal`)
- Add model evaluation report command in CLI
- Add Streamlit/Flask UI on top of the current logic


### Note 

THIS IS A FUN PROJECT , IT HAS NOTHING TO DO IF SOMEOINE HAS DIABETICS OR HEART DISEASES , DONOT REPLY ON THEN MODEL OUTPUT , PLEASE REFER A DOCTOR / SPECIALIST