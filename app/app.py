from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model & scaler
model = joblib.load(os.path.join("model", "student_model.pkl"))
scaler = joblib.load(os.path.join("model", "scaler.pkl"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['address']),
            float(request.form['Medu']),
            float(request.form['Dalc']),
            float(request.form['studytime']),
            float(request.form['failures']),
            float(request.form['G1']),
            float(request.form['G2']),
            float(request.form['goout'])  # Include "going out frequency"
        ]

        # Handle one-hot encoding for Pjob (must match training setup with drop_first=True)
        Pjob = request.form['Pjob']
        # Order: at_home, health, other, services (teacher is dropped as base case)
        Pjob_encoded = [0, 0, 0, 0]

        if Pjob == 'at_home':
            Pjob_encoded[0] = 1
        elif Pjob == 'health':
            Pjob_encoded[1] = 1
        elif Pjob == 'other':
            Pjob_encoded[2] = 1
        elif Pjob == 'services':
            Pjob_encoded[3] = 1
        # If Pjob is 'teacher', keep all 0s

        # Combine base features + encoded Pjob
        features.extend(Pjob_encoded)

        # Debugging: Print input features and their count
        print(f"Input features: {features}")
        print(f"Number of input features: {len(features)}")

        # Scale the input features
        scaled = scaler.transform([features])

        # Make prediction
        prediction = model.predict(scaled)[0]

        # Result: Pass or Fail
        result = "✅ Pass" if prediction == 1 else "❌ Fail"
        return render_template('index.html', result=result)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
