from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open("model/stroke_model.pkl", "rb"))

# Mappings for categorical features
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
work_type_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2}
residence_map = {"Urban": 1, "Rural": 0}
smoking_map = {"never smoked": 0, "smokes": 1, "formerly smoked": 2}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        gender = gender_map.get(request.form["gender"], 0)
        age = float(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        ever_married = married_map.get(request.form["ever_married"], 0)
        work_type = work_type_map.get(request.form["work_type"], 0)
        residence = residence_map.get(request.form["Residence_type"], 0)
        avg_glucose_level = float(request.form["avg_glucose_level"])
        bmi = float(request.form["bmi"])
        smoking_status = smoking_map.get(request.form["smoking_status"], 0)

        input_data = pd.DataFrame([[
            gender, age, hypertension, heart_disease, ever_married,
            work_type, residence, avg_glucose_level, bmi, smoking_status
        ]], columns=[
            "gender", "age", "hypertension", "heart_disease", "ever_married",
            "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"
        ])

        prediction = model.predict(input_data)[0]
        result = "⚠️ High Stroke Risk" if prediction == 1 else "✅ Low Stroke Risk"
        return render_template("index.html", prediction=result)

    return render_template("index.html", prediction=None)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
