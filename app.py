import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Load saved scaler and model
scaler = pickle.load(open("models/scaler.pkl","rb"))
model = pickle.load(open("models/ridge.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def predict():

    if request.method == "POST":

        # Get values from form
        Temp = float(request.form.get("Temp"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # Prepare data
        data = [[Temp, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]

        # Scale data
        scaled = scaler.transform(data)

        # Predict
        prediction = model.predict(scaled)

        value = prediction[0]

        # Risk level logic
        if value < 5:
            risk = "Low Risk"
        elif value < 15:
            risk = "Moderate Risk"
        elif value < 30:
            risk = "High Risk"
        else:
            risk = "Extreme Risk"

        return render_template(
            "home.html",
            result=round(value,2),
            risk=risk
        )

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)