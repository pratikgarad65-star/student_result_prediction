import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the pickle model
MODEL_PATH = "svm_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# HTML Template with Embedded CSS Styling and CSS Animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }

        body {
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #0284c7);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            padding: 20px;
            color: #fff;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .container {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 850px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            animation: fadeIn 1s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h2 {
            text-align: center;
            margin-bottom: 25px;
            font-weight: 700;
            letter-spacing: 1px;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.85rem;
            margin-bottom: 6px;
            color: #cbd5e1;
            font-weight: 600;
        }

        input, select {
            width: 100%;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        select option {
            background-color: #1e293b;
            color: #fff;
        }

        input:focus, select:focus {
            background: rgba(255, 255, 255, 0.15);
            border-color: #38bdf8;
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 15px;
            padding: 14px;
            background: linear-gradient(135deg, #0284c7, #6366f1);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
        }

        .btn-submit:active {
            transform: translateY(1px);
        }

        .result-card {
            margin-top: 30px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        @keyframes popIn {
            0% { transform: scale(0.8); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .result-success {
            background: rgba(34, 197, 94, 0.2);
            border: 1px solid #22c55e;
            color: #4ade80;
        }

        .result-danger {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid #ef4444;
            color: #f87171;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Student Performance Predictor</h2>
    
    <form method="POST" action="/predict">
        <div class="grid-container">
            <div class="form-group">
                <label>Gender</label>
                <select name="gender" required>
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>

            <div class="form-group">
                <label>Age</label>
                <input type="number" name="age" min="10" max="100" required placeholder="e.g. 18">
            </div>

            <div class="form-group">
                <label>Study Hours / Week</label>
                <input type="number" step="0.1" name="study_hours_per_week" required placeholder="e.g. 15.5">
            </div>

            <div class="form-group">
                <label>Attendance Rate (%)</label>
                <input type="number" step="0.1" name="attendance_rate" min="0" max="100" required placeholder="e.g. 85.0">
            </div>

            <div class="form-group">
                <label>Parent Education Level</label>
                <select name="parent_education" required>
                    <option value="0">High School</option>
                    <option value="1">Bachelor's</option>
                    <option value="2">Master's / Higher</option>
                </select>
            </div>

            <div class="form-group">
                <label>Internet Access</label>
                <select name="internet_access" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label>Extracurricular Activities</label>
                <select name="extracurricular" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label>Previous Score</label>
                <input type="number" step="0.1" name="previous_score" min="0" max="100" required placeholder="e.g. 75.0">
            </div>

            <div class="form-group">
                <label>Final Score</label>
                <input type="number" step="0.1" name="final_score" min="0" max="100" required placeholder="e.g. 80.0">
            </div>

            <button type="submit" class="btn-submit">Predict Outcome</button>
        </div>
    </form>

    {% if prediction %}
        <div class="result-card {% if prediction == 'Yes' %}result-success{% else %}result-danger{% endif %}">
            Prediction Result: <strong>{{ prediction }}</strong>
        </div>
    {% endif %}

    {% if error %}
        <div class="result-card result-danger">
            {{ error }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error="Model file not loaded properly on the server.")

    try:
        # Extract features from form input (strictly following feature order)
        features = [
            float(request.form.get("gender")),
            float(request.form.get("age")),
            float(request.form.get("study_hours_per_week")),
            float(request.form.get("attendance_rate")),
            float(request.form.get("parent_education")),
            float(request.form.get("internet_access")),
            float(request.form.get("extracurricular")),
            float(request.form.get("previous_score")),
            float(request.form.get("final_score"))
        ]

        # Reshape input for single-sample prediction
        input_data = np.array([features])
        
        # Predict using loaded model
        prediction_result = model.predict(input_data)[0]

        return render_template_string(HTML_TEMPLATE, prediction=prediction_result)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
