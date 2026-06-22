# FLASK APPLICATION
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)
app = application

# Import the ridge regressor and standard scaler pickle files.
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

with (MODELS_DIR / "ridge.pkl").open("rb") as ridge_file:
    ridge_model = pickle.load(ridge_file)

with (MODELS_DIR / "scaler.pkl").open("rb") as scaler_file:
    standard_scaler = pickle.load(scaler_file)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        input_data = pd.DataFrame(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]],
            columns=[
                "Temperature",
                "RH",
                "Ws",
                "Rain",
                "FFMC",
                "DMC",
                "ISI",
                "Classes",
                "Region",
            ],
        )

        new_data_scaled = standard_scaler.transform(input_data)
        result = ridge_model.predict(new_data_scaled)

        return render_template("home.html", result=round(float(result[0]), 2))
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
