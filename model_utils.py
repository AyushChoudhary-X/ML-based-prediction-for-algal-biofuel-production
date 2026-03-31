from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import VarianceThreshold

import pandas as pd
import numpy as np

# ================= TRAINING =================
def train_all_models(df, target, selected_models):

    # ---------------- 1. SEPARATE TARGET & FEATURES ----------------
    # Drop rows where the target column is missing to prevent fit() crashes
    df = df.dropna(subset=[target]).copy()
    
    X_full = df.drop(columns=[target])
    y = df[target]

    # If the target is categorical (text), convert to numerical for regressors
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # ---------------- 2. HANDLE MISSING VALUES & CATEGORICAL DATA ----------------
    # Separate numeric and categorical columns
    num_cols = X_full.select_dtypes(include=np.number).columns
    cat_cols = X_full.select_dtypes(exclude=np.number).columns

    # Fill missing numbers with the mean, and missing text with 'Unknown'
    X_full[num_cols] = X_full[num_cols].fillna(X_full[num_cols].mean())
    X_full[cat_cols] = X_full[cat_cols].fillna('Unknown')

    # Convert text columns to numbers (One-Hot Encoding)
    X_encoded = pd.get_dummies(X_full, columns=cat_cols, drop_first=True)

    # ---------------- 3. VARIANCE FILTER ----------------
    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X_encoded)

    selected_features = X_encoded.columns[selector.get_support()]
    X = pd.DataFrame(X_var, columns=selected_features)

    # ---------------- 4. SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- 5. SCALING ----------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- 6. MODELS ----------------
    all_models = {
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_split=2, min_samples_leaf=2),
        "SVR": SVR(),
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(),
        "ANN": MLPRegressor(max_iter=500)
    }

    models = {k: all_models[k] for k in selected_models}

    # ---------------- 7. STORAGE ----------------
    results = []
    trained_models = {}
    predictions = {}
    feature_importance = {}

    # ---------------- 8. TRAIN LOOP ----------------
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        results.append({
            "name": name,
            "r2": r2,
            "rmse": rmse
        })

        trained_models[name] = model
        predictions[name] = pred

        # -------- FEATURE IMPORTANCE --------
        if hasattr(model, "feature_importances_"):
            feature_importance[name] = model.feature_importances_
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            if coefs.ndim > 1:
                coefs = np.mean(coefs, axis=0)
            feature_importance[name] = coefs
        else:
            # Fallback for models without feature importance (like SVR, KNN)
            feature_importance[name] = np.zeros(X.shape[1])

    # ---------------- 9. BEST MODEL ----------------
    best_model = max(results, key=lambda x: x["r2"])["name"]

    return {
        "models": trained_models,
        "results": results,
        "best_model": best_model,
        "y_test": pd.Series(y_test).reset_index(drop=True),
        "predictions": predictions,
        "feature_names": list(X.columns),
        "scaler": scaler,
        "X_test": X_test.reset_index(drop=True),
        "X": X,  
        "feature_importance": feature_importance
    }

# ================= PREDICTION =================
def predict_best(input_dict, data):
    model = data["models"][data["best_model"]]
    scaler = data["scaler"]
    X = data["X"]
    features = data["feature_names"]

    df = pd.DataFrame([input_dict])

    # Ensure correct column order
    df = df.reindex(columns=features)

    # Fill missing values
    for col in df.columns:
        if df[col].iloc[0] == "" or pd.isna(df[col].iloc[0]):
            df[col] = X[col].mean()
        else:
            df[col] = float(df[col])

    df_scaled = scaler.transform(df)
    return round(model.predict(df_scaled)[0], 4)

# ================= OPTIMIZATION =================
def optimize_inputs(data, mode="max", n_iter=1000):
    model = data["models"][data["best_model"]]
    scaler = data["scaler"]
    X = data["X"]
    features = data["feature_names"]

    if mode == "max":
        best_output = -np.inf
    else:
        best_output = np.inf

    best_input = None

    for _ in range(n_iter):
        sample = []
        for col in features:
            val = np.random.uniform(X[col].min(), X[col].max())
            sample.append(val)

        sample_array = np.array(sample).reshape(1, -1)
        sample_scaled = scaler.transform(sample_array)
        pred = model.predict(sample_scaled)[0]

        if mode == "max":
            if pred > best_output:
                best_output = pred
                best_input = sample
        else:
            if pred < best_output:
                best_output = pred
                best_input = sample

    best_input_dict = dict(zip(features, best_input))
    return best_input_dict, round(best_output, 4)