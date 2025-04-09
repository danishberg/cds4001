import os
import glob
import zipfile
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
import kagglehub  # This assumes kagglehub is available via pip

app = Flask(__name__)
CORS(app)

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def extract_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

# Download and extract the Sleep and Health Metrics dataset
metrics_zip_file = os.path.join(DATA_DIR, 'sleep_health_metrics.zip')
if not os.path.exists(metrics_zip_file):
    metrics_zip_file = kagglehub.dataset_download("uom190346a/sleep-and-health-metrics")
metrics_extract_dir = os.path.join(DATA_DIR, 'metrics')
if not os.path.exists(metrics_extract_dir):
    os.makedirs(metrics_extract_dir)
    extract_zip(metrics_zip_file, metrics_extract_dir)

# Download and extract the Sleep Health and Lifestyle dataset
lifestyle_zip_file = os.path.join(DATA_DIR, 'sleep_health_lifestyle.zip')
if not os.path.exists(lifestyle_zip_file):
    lifestyle_zip_file = kagglehub.dataset_download("henryshan/sleep-health-and-lifestyle")
lifestyle_extract_dir = os.path.join(DATA_DIR, 'lifestyle')
if not os.path.exists(lifestyle_extract_dir):
    os.makedirs(lifestyle_extract_dir)
    extract_zip(lifestyle_zip_file, lifestyle_extract_dir)

# Load CSV files from extracted directories
metrics_csv_files = glob.glob(os.path.join(metrics_extract_dir, "*.csv"))
lifestyle_csv_files = glob.glob(os.path.join(lifestyle_extract_dir, "*.csv"))

if metrics_csv_files:
    df_metrics = pd.read_csv(metrics_csv_files[0])
else:
    df_metrics = pd.DataFrame()

if lifestyle_csv_files:
    df_lifestyle = pd.read_csv(lifestyle_csv_files[0])
else:
    df_lifestyle = pd.DataFrame()

# Merge datasets using 'UserID' if available or concatenate otherwise
if 'UserID' in df_metrics.columns and 'UserID' in df_lifestyle.columns:
    df_merged = pd.merge(df_metrics, df_lifestyle, on='UserID', how='inner')
else:
    df_merged = pd.concat([df_metrics, df_lifestyle], axis=1)

# Create dummy columns if missing
if 'sleep_duration' not in df_merged.columns:
    df_merged['sleep_duration'] = np.random.uniform(4, 10, size=len(df_merged))
if 'exercise_freq' not in df_merged.columns:
    df_merged['exercise_freq'] = np.random.randint(0, 8, size=len(df_merged))

# Generate risk label: 2 for High (<6h), 1 for Moderate (6-8h), 0 for Low (>8h)
def generate_label(row):
    if row['sleep_duration'] < 6:
        return 2
    elif row['sleep_duration'] <= 8:
        return 1
    else:
        return 0

df_merged['risk_label'] = df_merged.apply(generate_label, axis=1)

# Train logistic regression model using sleep_duration and exercise_freq
features = ['sleep_duration', 'exercise_freq']
X_train = df_merged[features].values
y_train = df_merged['risk_label'].values
model = LogisticRegression(multi_class='multinomial', max_iter=200)
model.fit(X_train, y_train)

def classify_label(label: int):
    mapping = {0: "Low", 1: "Moderate", 2: "High"}
    return mapping.get(label, "Unknown")

# Prediction endpoint for individual scenarios
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sleep_hours = data.get('sleepHours', 7)
    exercise_freq = data.get('exerciseFreq', 3)
    features_input = np.array([[sleep_hours, exercise_freq]])
    pred = model.predict(features_input)[0]
    prob = model.predict_proba(features_input)[0]
    confidence = float(np.max(prob))
    risk_score = int(confidence * 100)
    return jsonify({
        'risk_score': risk_score,
        'classification': classify_label(pred),
        'confidence': round(confidence, 2)
    })

# Simulation endpoint for multiple scenarios
@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    scenarios = data.get('scenarios', [])
    results = []
    for scenario in scenarios:
        sleep_hours = scenario.get('sleepHours', 7)
        exercise_freq = scenario.get('exerciseFreq', 3)
        features_input = np.array([[sleep_hours, exercise_freq]])
        pred = model.predict(features_input)[0]
        prob = model.predict_proba(features_input)[0]
        confidence = float(np.max(prob))
        risk_score = int(confidence * 100)
        results.append({
            'sleepHours': sleep_hours,
            'exerciseFreq': exercise_freq,
            'risk_score': risk_score,
            'classification': classify_label(pred),
            'confidence': round(confidence, 2)
        })
    return jsonify(results)

# Summary endpoint for overall dataset statistics
@app.route('/summary', methods=['GET'])
def summary():
    if df_merged.empty:
        return jsonify({"error": "No data available"}), 404
    summary_stats = {
        "sleep_duration": {
            "mean": df_merged['sleep_duration'].mean(),
            "min": df_merged['sleep_duration'].min(),
            "max": df_merged['sleep_duration'].max()
        },
        "exercise_freq": {
            "mean": df_merged['exercise_freq'].mean(),
            "min": df_merged['exercise_freq'].min(),
            "max": df_merged['exercise_freq'].max()
        },
        "risk_distribution": df_merged['risk_label'].value_counts().to_dict()
    }
    return jsonify(summary_stats)

if __name__ == '__main__':
    app.run(debug=True)
