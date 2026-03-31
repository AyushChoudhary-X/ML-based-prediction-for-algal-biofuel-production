from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from model_utils import train_all_models, predict_best, optimize_inputs
import matplotlib.pyplot as plt
import io
import base64
import plotly
import shap
import plotly.graph_objs as go
import json


def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


app = Flask(__name__)

data_store = {}


@app.route('/')
def home():
    return render_template("index.html", active='index')

# -------- Upload CSV --------


@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['file']
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    data_store["df"] = df

    return render_template(
        "train.html",
        active="train",
        columns=df.columns,
        preview=df.head().to_html(classes="table", index=False)
    )

# -------- Train --------


@app.route('/train', methods=['GET', 'POST'])
def train():

    # If user comes from sidebar (GET request)
    if request.method == 'GET':

        if "df" not in data_store:
            return redirect('/')

        return render_template(
            "train.html",
            active="train",
            columns=data_store["df"].columns,
            preview=data_store["df"].head().to_html(
                classes="table", index=False)
        )

    # If form is submitted (POST request)
    target = request.form['target']
    selected_models = request.form.getlist("models")

    df = data_store["df"]

    results = train_all_models(df, target, selected_models)

    data_store.clear()
    data_store.update(results)
    data_store["df"] = df

    from flask import url_for

    return redirect(url_for('dashboard'))

# -------- Dashboard --------


@app.route('/dashboard')
def dashboard():

    if "results" not in data_store:
        return redirect('/')

    results = data_store["results"]
    best = data_store["best_model"]

    # -------- DATA --------
    names = [r["name"] for r in results]
    r2_scores = [r["r2"] for r in results]
    rmse_scores = [r["rmse"] for r in results]

    # -------- R2 CHART --------
    fig_r2 = go.Figure()

    fig_r2.add_trace(go.Bar(
        x=names,
        y=r2_scores,
        text=[round(x, 3) for x in r2_scores],
        textposition='auto'
    ))

    fig_r2.update_layout(
        title="Model Comparison (R² Score)",
        xaxis_title="Models",
        yaxis_title="R² Score"
    )

    chart_r2 = json.dumps(fig_r2, cls=plotly.utils.PlotlyJSONEncoder)

    # -------- RMSE CHART --------
    fig_rmse = go.Figure()

    fig_rmse.add_trace(go.Bar(
        x=names,
        y=rmse_scores,
        text=[round(x, 3) for x in rmse_scores],
        textposition='auto'
    ))

    fig_rmse.update_layout(
        title="Model Comparison (RMSE)",
        xaxis_title="Models",
        yaxis_title="RMSE"
    )

    chart_rmse = json.dumps(fig_rmse, cls=plotly.utils.PlotlyJSONEncoder)

    # -------- SCATTER --------
    y_test = data_store["y_test"]
    preds = data_store["predictions"][best]

    fig_scatter = go.Figure()

    fig_scatter.add_trace(go.Scatter(
        x=y_test,
        y=preds,
        mode='markers',
        name='Predictions'
    ))

    fig_scatter.add_trace(go.Scatter(
        x=[min(y_test), max(y_test)],
        y=[min(y_test), max(y_test)],
        mode='lines',
        name='Ideal',
        line=dict(dash='dash')
    ))

    fig_scatter.update_layout(
        title=f"Actual vs Predicted ({best})",
        xaxis_title="Actual",
        yaxis_title="Predicted"
    )

    chart_scatter = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "dashboard.html",
        active="dashboard",
        results=results,
        best=best,
        chart_r2=chart_r2,
        chart_rmse=chart_rmse,
        chart_scatter=chart_scatter
    )

    # ---------------- SCATTER PLOT ----------------
    y_test = data_store["y_test"]
    preds = data_store["predictions"][best]

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, preds)

    min_val = min(min(y_test), min(preds))
    max_val = max(max(y_test), max(preds))

    ax2.plot([min_val, max_val], [min_val, max_val],
             linestyle='--', color='red')

    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")

    chart2 = plot_to_base64(fig2)

    return render_template(
        "dashboard.html",
        active="dashboard",
        results=results,
        best=best,
        chart1=chart1,
        chart2=chart2
    )


# -------- Predict --------

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if "models" not in data_store:
        return "⚠️ Please train models first"

    features = data_store["feature_names"]
    scaler = data_store["scaler"]
    model = data_store["models"][data_store["best_model"]]

    prediction = None

    if request.method == 'POST':
        features = data_store["feature_names"]
        scaler = data_store["scaler"]
        model = data_store["models"][data_store["best_model"]]
        
        # USE THE ENCODED 'X' DATAFRAME INSTEAD OF 'df_original'
        df_encoded = data_store["X"] 

        input_data = []
        filled_data = {}

        for f in features:
            val = request.form.get(f)

            if val == "" or val is None:
                # AUTO FILL USING MEAN FROM ENCODED DATA
                mean_val = df_encoded[f].mean()
                input_data.append(mean_val)
                filled_data[f] = round(mean_val, 4)
            else:
                val = float(val)
                input_data.append(val)
                filled_data[f] = val

        import numpy as np
        input_array = np.array(input_data).reshape(1, -1)

        # Scale
        input_scaled = scaler.transform(input_array)

        prediction = round(model.predict(input_scaled)[0], 4)

        # STORE CLEAN DATA FOR DOWNLOAD
        data_store["last_input"] = filled_data
        data_store["last_prediction"] = prediction

    return render_template(
        "predict.html",
        active="predict",
        features=features,
        prediction=prediction,
        best=data_store["best_model"]
    )

# -------- Download results--------

from flask import Response

@app.route('/download_results')
def download_results():

    import pandas as pd

    if "y_test" not in data_store:
        return "No results available"

    y_test = data_store["y_test"].reset_index(drop=True)
    X_test = pd.DataFrame(
        data_store["X_test"],
        columns=data_store["feature_names"]
    ).reset_index(drop=True)

    preds = data_store["predictions"]

    # Combine everything
    df = X_test.copy()
    df["Actual"] = y_test

    for model_name, values in preds.items():
        df[f"{model_name}_Predicted"] = values

    csv = df.to_csv(index=False)

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=full_results.csv"}
    )

# -------- Download predictions--------

from flask import Response

@app.route('/download_prediction')
def download_prediction():

    import pandas as pd

    if "last_prediction" not in data_store:
        return "No prediction available"

    input_data = data_store["last_input"]
    prediction = data_store["last_prediction"]

    df = pd.DataFrame([input_data])
    df["Prediction"] = prediction

    csv = df.to_csv(index=False)

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=prediction.csv"}
    )

# -------- Featurr Importance --------


@app.route('/feature_importance')
def feature_importance():

    if "feature_importance" not in data_store:
        return redirect('/')

    features = data_store["feature_names"]
    fi_data = data_store["feature_importance"]

    charts = {}

    for model_name, values in fi_data.items():

        # Sort values
        import numpy as np
        idx = np.argsort(values)

        sorted_features = [features[i] for i in idx]
        sorted_values = values[idx]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=sorted_values,
            y=sorted_features,
            orientation='h'
        ))

        fig.update_layout(
            title=model_name,
            margin=dict(l=150),
            height=400
        )

        charts[model_name] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "feature_importance.html",
        active="feature_importance",
        charts=charts
    )


# -------- Optimization --------

@app.route('/optimize', methods=['GET', 'POST'])
def optimize():

    if "models" not in data_store:
        return redirect('/')

    mode = "max"  # default

    if request.method == "POST":
        mode = request.form.get("mode")

    best_inputs, best_output = optimize_inputs(data_store, mode=mode)

    return render_template(
        "optimize.html",
        inputs=best_inputs,
        output=best_output,
        mode=mode,
        active="optimize"
    )

# -------- Shap --------
@app.route('/shap')
def shap_explain():

    if "models" not in data_store:
        return redirect('/')

    model = data_store["models"][data_store["best_model"]]
    X = data_store["X"]
    feature_names = data_store["feature_names"]

    # Use small sample (fast)
    X_sample = X.sample(min(100, len(X)), random_state=42)

    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    # Plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    img = base64.b64encode(buf.getvalue()).decode()

    return render_template("shap.html", shap_plot=img, active="shap")

# -------- About --------
@app.route('/about')
def about():
    return render_template("about.html", active="about")


if __name__ == "__main__":
    app.run(debug=True)
