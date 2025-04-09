import os
import glob
import zipfile
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample  # For balancing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define features and target
MODEL_FEATURES = ['sleep_duration', 'exercise_freq', 'Age', 'Gender_encoded', 'Stress Level']
TARGET_VARIABLE = 'risk_label'

# Global variables to store model and related info
df_summary_global = None  # For summary/correlation endpoints
pipeline_global = None    # Trained model pipeline
model_accuracy_global = None  # Model accuracy
feature_names_global = None   # List of features used

# --- Helper Functions ---
def generate_risk_label(sleep_duration):
    """
    Generate a risk label based on sleep duration.
    Labels: 0 = Extremely Low risk, 1 = Very Low, 2 = Low, 3 = Moderate,
            4 = High, 5 = Very High, 6 = Extreme
    """
    if pd.isna(sleep_duration): 
        return 3  # Default to Moderate if missing
    if sleep_duration < 4.5:
        return 6     # Extreme risk
    elif sleep_duration < 5.5:
        return 5     # Very High
    elif sleep_duration < 6.5:
        return 4     # High
    elif sleep_duration < 7.5:
        return 3     # Moderate
    elif sleep_duration < 8.5:
        return 2     # Low
    elif sleep_duration < 9.5:
        return 1     # Very Low
    else:
        return 0     # Extremely Low

def classify_risk_label(label: int) -> str:
    mapping = {0: "Extremely Low", 1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High", 6: "Extreme"}
    return mapping.get(label, "Unknown")

def load_and_prepare_data():
    global feature_names_global
    logging.info("Loading and preparing data from local file data/data2.csv...")

    data_dir = 'data'
    file_path = os.path.join(data_dir, 'data2.csv')

    if not os.path.exists(data_dir):
        logging.error(f"Data directory '{data_dir}' not found. Please create it.")
        return None, None, None 
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}. Please ensure data2.csv is in the '{data_dir}' folder.")
        return None, None, None 

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path} with shape {df.shape}")
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return None, None, None

    # Rename columns for uniformity
    df.rename(columns={
        'Sleep Duration': 'sleep_duration',
        'Physical Activity Level': 'exercise_freq',
    }, inplace=True)

    # Convert key columns to numeric
    for col in ['sleep_duration', 'exercise_freq', 'Age', 'Stress Level', 'Heart Rate', 'Daily Steps', 'Quality of Sleep']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logging.warning(f"Column '{col}' not found.")

    # Impute missing values using median
    for col in ['sleep_duration', 'exercise_freq', 'Age', 'Stress Level']:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logging.info(f"Imputed missing {col} with median value {median_val}")
        else:
            logging.error(f"Critical feature '{col}' missing.")
            return None, None, None

    # Encode Gender
    if 'Gender' in df.columns:
        le = LabelEncoder()
        df['Gender_encoded'] = le.fit_transform(df['Gender'].astype(str).fillna('Unknown'))
        logging.info("Encoded 'Gender' into 'Gender_encoded'")
    else:
        logging.error("Column 'Gender' missing.")
        return None, None, None

    # Check required features
    missing_features = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing_features:
         logging.error(f"Missing required features: {missing_features}")
         return None, None, None

    # Generate risk labels
    df[TARGET_VARIABLE] = df['sleep_duration'].apply(generate_risk_label)
    logging.info("Generated risk labels.")

    # Balance the dataset by oversampling each class equally
    counts = df[TARGET_VARIABLE].value_counts()
    max_count = counts.max()
    df_list = []
    for cls in df[TARGET_VARIABLE].unique():
        df_cls = df[df[TARGET_VARIABLE] == cls]
        df_cls_resampled = resample(df_cls, replace=True, n_samples=max_count, random_state=42)
        df_list.append(df_cls_resampled)
    df_balanced = pd.concat(df_list).sample(frac=1, random_state=42)
    df = df_balanced

    df.dropna(subset=[TARGET_VARIABLE], inplace=True)

    # --- Model Training ---
    try:
        X = df[MODEL_FEATURES]
        y = df[TARGET_VARIABLE]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        numeric_features = ['sleep_duration', 'exercise_freq', 'Age', 'Stress Level']
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), numeric_features)],
            remainder='passthrough'
        )

        # Use multinomial logistic regression with lbfgs solver
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        logging.info("Trained logistic regression pipeline.")

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model accuracy: {accuracy:.3f}")

        feature_names_global = MODEL_FEATURES
        return df, pipeline, accuracy
    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        return df, None, None

# Load data and train model on startup
df_summary_global, pipeline_global, model_accuracy_global = load_and_prepare_data()

# --- API Endpoints ---
@app.route('/model-info', methods=['GET'])
def model_info():
    if model_accuracy_global is None:
        return jsonify({"error": "Model accuracy not available."}), 503
    return jsonify({
        "accuracy": round(model_accuracy_global, 3),
        "features_used": feature_names_global,
        "target": TARGET_VARIABLE,
        "model_type": "Logistic Regression (Pipeline)"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline_global is None:
        logging.error("Model pipeline not available.")
        abort(503, description="Model not loaded.")
    if not request.is_json:
        abort(400, description="Request must be JSON.")

    data = request.get_json()
    is_simulation = 'age' not in data  # Heuristic to distinguish simulation

    try:
        sleep_hours = float(data.get('sleepHours'))
        exercise_freq = int(data.get('exerciseFreq'))
        age = int(data.get('age', 35))
        gender = str(data.get('gender', 'Female'))
        stress_level = int(data.get('stressLevel', 5))

        # Validate types (basic)
        if not all(isinstance(x, (int, float)) for x in [sleep_hours, exercise_freq, age, stress_level]):
            raise ValueError("Numeric inputs required.")
        if gender not in ['Male', 'Female', 'Other']:
            logging.warning(f"Gender '{gender}' not standard, defaulting to 'Female'.")
            gender = 'Female'
    except Exception as e:
        logging.error(f"Invalid input data: {e} - {data}")
        abort(400, description=f"Invalid input: {e}")

    try:
        # Encode gender consistently (assuming Female=0, Male=1 as used during training)
        gender_encoded = 1 if gender == 'Male' else 0

        # Create DataFrame for prediction
        input_df = pd.DataFrame([[sleep_hours, exercise_freq, age, gender_encoded, stress_level]], columns=MODEL_FEATURES)

        # Get probabilities and predicted label from the pipeline
        pred_proba = pipeline_global.predict_proba(input_df)[0]
        pred_label = int(np.argmax(pred_proba))

        # --- NEW: Compute Continuous Risk Score ---
        # Compute the weighted average over class labels (0 to 6)
        weighted_label = sum(prob * label for label, prob in enumerate(pred_proba))
        # Normalize to percentage: if maximum label is 6, then:
        risk_score_cont = (weighted_label / 6) * 100

        classification = classify_risk_label(pred_label)
        confidence = float(np.max(pred_proba))

        logging.info(f"Prediction: Input=(sleep={sleep_hours}, exercise={exercise_freq}, age={age}, gender={gender}, stress={stress_level}), "
                     f"Weighted Label={weighted_label:.3f}, Risk Score={risk_score_cont:.1f}%, Predicted Label={pred_label}, Classification={classification}")
        return jsonify({
            'risk_score': round(risk_score_cont, 1),  # Continuous risk score percentage
            'classification': classification,
            'confidence': round(confidence, 3)
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        abort(500, description="Internal server error during prediction.")

@app.route('/summary', methods=['GET'])
def summary():
    if df_summary_global is None or df_summary_global.empty:
        logging.warning("No data for summary.")
        abort(404, description="Data not available for summary.")
    try:
        summary_stats = {}
        if 'sleep_duration' in df_summary_global.columns:
            summary_stats["Sleep Duration (hours)"] = {
                "mean": round(df_summary_global['sleep_duration'].mean(), 2),
                "median": round(df_summary_global['sleep_duration'].median(), 2),
                "min": round(df_summary_global['sleep_duration'].min(), 2),
                "max": round(df_summary_global['sleep_duration'].max(), 2)
            }
        if 'exercise_freq' in df_summary_global.columns:
            summary_stats["Exercise Frequency (days/wk)"] = {
                "mean": round(df_summary_global['exercise_freq'].mean(), 2),
                "median": round(df_summary_global['exercise_freq'].median(), 2),
                "min": round(df_summary_global['exercise_freq'].min(), 2),
                "max": round(df_summary_global['exercise_freq'].max(), 2)
            }
        if TARGET_VARIABLE in df_summary_global.columns:
            risk_counts = df_summary_global[TARGET_VARIABLE].map(classify_risk_label).value_counts().to_dict()
            summary_stats["Predicted Risk Distribution"] = risk_counts

        logging.info("Generated summary statistics.")
        return jsonify(summary_stats)
    except Exception as e:
        logging.error(f"Error generating summary: {e}", exc_info=True)
        abort(500, description="Internal server error during summary generation.")

@app.route('/correlation', methods=['GET'])
def correlation():
    if df_summary_global is None or df_summary_global.empty:
        logging.warning("No data available for correlation analysis.")
        abort(404, description="Data not available for correlation analysis.")
    try:
        numeric_df = df_summary_global.select_dtypes(include=np.number)
        numeric_cols = numeric_df.columns.tolist()
        logging.info(f"Numeric columns for correlation: {numeric_cols}")

        cols_to_exclude = [col for col in numeric_cols if '_encoded' in col or 'ID' in col or 'Id' in col]
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
        logging.info(f"Numeric columns after exclusion: {numeric_cols}")

        if not numeric_cols:
            logging.error("No numeric columns for correlation analysis.")
            abort(400, description="No numeric columns found.")
        correlation_matrix = df_summary_global[numeric_cols].corr()
        logging.info(f"Correlation matrix shape: {correlation_matrix.shape}")

        if correlation_matrix.isnull().all().all():
             logging.warning("Correlation matrix all NaN.")
             abort(500, description="Could not calculate correlations.")
        correlation_data = correlation_matrix.fillna('N/A').reset_index().to_dict(orient='records')
        response_data = {'columns': numeric_cols, 'correlation': correlation_data }
        logging.info("Correlation matrix generated.")
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Error during correlation analysis: {e}", exc_info=True)
        abort(500, description="Internal server error during correlation analysis.")

# --- Error Handlers ---
@app.errorhandler(400)
def bad_request(error):
    return jsonify(error=str(error.description)), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify(error=str(error.description)), 404

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify(error=str(error.description)), 500

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify(error=str(error.description)), 503

# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
