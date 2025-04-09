# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

X_train = np.array([
    [7, 3],
    [6, 2],
    [8, 4],
    [5, 1],
    [7, 3],
    [9, 5],
    [4, 1],
    [6, 2]
])
y_train = np.array([1, 2, 0, 2, 1, 0, 2, 2])

model = LogisticRegression(multi_class='multinomial', max_iter=200)
model.fit(X_train, y_train)

def classify_label(label: int):
    mapping = {0: "Low", 1: "Moderate", 2: "High"}
    return mapping.get(label, "Unknown")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sleep_hours = data.get('sleepHours', 7)
    exercise_freq = data.get('exerciseFreq', 3)
    features = np.array([[sleep_hours, exercise_freq]])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    confidence = float(np.max(prob))
    classification = classify_label(pred)
    risk_score = int(confidence * 100)
    return jsonify({
        'risk_score': risk_score,
        'classification': classification,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
