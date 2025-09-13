from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import os

app = Flask(__name__)
MODEL_FILE = "model.pkl"

# Train model if not exists
if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("sample_data.csv")
    X = df[["bed_occupancy","active_cases","staff_on_duty","previous_oxygen_usage"]]
    y = df["oxygen_demand"]
    model = LinearRegression().fit(X, y)
    dump(model, MODEL_FILE)
else:
    model = load(MODEL_FILE)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Example JSON:
    {
      "bed_occupancy":80,
      "active_cases":35,
      "staff_on_duty":12,
      "previous_oxygen_usage":200
    }
    """
    data = request.get_json()
    X_new = pd.DataFrame([data])
    y_pred = model.predict(X_new)[0]
    return jsonify({"predicted_oxygen_demand": round(y_pred, 2)})

if __name__ == "__main__":
    app.run(debug=True)
