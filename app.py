from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load models
decision_tree = joblib.load("models/Decision_tree_model.pkl")
svr_model = joblib.load("models/SVR_model.pkl")
kmeans_model = joblib.load("models/kmeans_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')  # UI form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        co = float(request.form['co'])
        ozone = float(request.form['ozone'])
        no2 = float(request.form['no2'])
        pm25 = float(request.form['pm25'])
        model_choice = request.form['model']

        # Prepare input
        input_data = np.array([[co, ozone, no2, pm25]])

        # Apply selected model
        if model_choice == "decision_tree":
            result = decision_tree.predict(input_data)[0]
            output = f"Decision Tree Prediction: {result:.2f}"

        elif model_choice == "svr":
            result = svr_model.predict(input_data)[0]
            output = f"SVR Prediction: {result:.2f}"

        elif model_choice == "kmeans":
            result = kmeans_model.predict(input_data)[0]
            output = f"KMeans Cluster: {int(result)}"

        else:
            output = "Invalid model selected."

        return render_template("index.html", prediction_text=output)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
