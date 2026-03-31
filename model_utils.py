from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import VarianceThreshold

import pandas as pd
import numpy as np

# ================= TRAINING =================
def train_all_models(df, target, selected_models):

    # 1. FILTER FOR ONLY ONE METRIC (e.g., Lipid Productivity - LP)
    # The paper trains 3 separate models. We will isolate the LP rows (HLP and LLP)
    df = df.dropna(subset=[target]).copy()
    df = df[df[target].isin(['HLP', 'LLP'])] # Only keep High LP and Low LP rows
    
    # 2. BINARY ENCODING
    # HLP (High Lipid Productivity) = 1, LLP (Low Lipid Productivity) = 0
    y = df[target].map({'HLP': 1, 'LLP': 0})
    X_full = df.drop(columns=[target])

    # 3. HANDLE MISSING VALUES & CATEGORICAL DATA
    num_cols = X_full.select_dtypes(include=np.number).columns
    cat_cols = X_full.select_dtypes(exclude=np.number).columns

    X_full[num_cols] = X_full[num_cols].fillna(X_full[num_cols].mean())
    X_full[cat_cols] = X_full[cat_cols].fillna('Unknown')

    X_encoded = pd.get_dummies(X_full, columns=cat_cols, drop_first=True)

    # 4. VARIANCE FILTER
    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X_encoded)

    selected_features = X_encoded.columns[selector.get_support()]
    X = pd.DataFrame(X_var, columns=selected_features)

    # 5. SPLIT 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. SCALING
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. CLASSIFICATION MODELS (Using the exact Hyperparameters from Table 3)
    all_models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=2, min_samples_leaf=2, random_state=42),
        "SVM": SVC(probability=True),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "ANN": MLPClassifier(max_iter=500)
    }

    models = {k: all_models[k] for k in selected_models}

    results = []
    trained_models = {}

    # 8. TRAIN LOOP
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

        # Use Classification Metrics instead of R2 and RMSE
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        results.append({
            "name": name,
            "accuracy": acc, # Replaced R2
            "f1_score": f1   # Replaced RMSE
        })

        trained_models[name] = model

    # 9. BEST MODEL
    best_model = max(results, key=lambda x: x["accuracy"])["name"]

    return {
        "models": trained_models,
        "results": results,
        "best_model": best_model,
        "scaler": scaler,
        "feature_names": list(X.columns)
    }

# ================= LOCAL TESTING BLOCK =================
if __name__ == "__main__":
    import pandas as pd

    # 1. Load your dataset (Ensure the CSV is in the same folder as this script)
    csv_filename = "1-s2.0-S0960148125015654-mmc2.xlsx - Sheet1.csv"
    
    print(f"Loading data from {csv_filename}...")
    try:
        df = pd.read_csv(csv_filename)
        df.columns = df.columns.str.strip() # Clean column names like in app.py
        
        # 2. Define the target column exactly as it appears in your dataset
        target_col = "OUTCOME High/Low LC, LP, BP"
        
        # 3. Pick a few models to test (matching the dictionary keys in your code)
        test_models = ["Random Forest", "Linear Regression", "XGBoost"]
        
        print(f"Starting training for models: {test_models}...")
        
        # 4. Call your function
        output_data = train_all_models(df, target=target_col, selected_models=test_models)
        
        # 5. Print out the results to verify it worked
        print("\n✅ --- TRAINING SUCCESSFUL --- ✅")
        print(f"Best Model Found: {output_data['best_model']}")
        
        print("\nIndividual Model Performance:")
        for res in output_data['results']:
            print(f"  -> {res['name']}: R² = {res['r2']:.4f} | RMSE = {res['rmse']:.4f}")
            
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file '{csv_filename}'. Make sure it is in the same folder.")
    except Exception as e:
        print(f"❌ An error occurred during training: {e}")