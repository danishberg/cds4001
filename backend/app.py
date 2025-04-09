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
import kagglehub
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

DATA_DIR = 'data'
MODEL_FEATURES = ['sleep_duration', 'exercise_freq']
TARGET_VARIABLE = 'risk_label'

# --- Data Loading and Preprocessing ---
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def extract_zip(zip_path, extract_dir):
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
            logging.info(f"Extracted {zip_path} to {extract_dir}")
    except zipfile.BadZipFile:
        logging.error(f"Error: {zip_path} is not a valid zip file or is corrupted.")
        return False
    except Exception as e:
        logging.error(f"Error extracting zip file {zip_path}: {e}")
        return False
    return True

def download_and_extract_dataset(dataset_slug, zip_filename, extract_subfolder):
    ensure_dir_exists(DATA_DIR)
    zip_path = os.path.join(DATA_DIR, zip_filename)
    extract_dir = os.path.join(DATA_DIR, extract_subfolder)

    if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
        logging.info(f"Downloading dataset: {dataset_slug}")
        try:
            zip_path = kagglehub.dataset_download(dataset_slug, path=DATA_DIR)
            logging.info(f"Downloaded {dataset_slug} to {zip_path}")
            ensure_dir_exists(extract_dir)
            if not extract_zip(zip_path, extract_dir):
                 # If extraction fails, remove the potentially corrupted dir
                 if os.path.exists(extract_dir):
                     os.rmdir(extract_dir) # Use shutil.rmtree if dir might not be empty
                 return None
        except Exception as e:
            logging.error(f"Failed to download or extract {dataset_slug}: {e}")
            return None
    else:
        logging.info(f"Dataset {dataset_slug} already exists in {extract_dir}")

    csv_files = glob.glob(os.path.join(extract_dir, "*.csv"))
    if not csv_files:
        logging.warning(f"No CSV files found in {extract_dir}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_files[0])
        logging.info(f"Loaded data from {csv_files[0]}")
        return df
    except Exception as e:
        logging.error(f"Error reading CSV {csv_files[0]}: {e}")
        return pd.DataFrame()

def generate_risk_label(sleep_duration):
    if pd.isna(sleep_duration):
        return 1 # Default to moderate if missing
    if sleep_duration < 6:
        return 2  # High risk
    elif sleep_duration <= 8:
        return 1  # Moderate risk
    else:
        return 0  # Low risk

def load_and_prepare_data():
    logging.info("Loading and preparing data from local file data2.csv...")
    
    data_dir = 'data' 
    file2_path = os.path.join(data_dir, 'data2.csv')
    
    # --- Load only data2.csv --- 
    try:
        df_merged = pd.read_csv(file2_path) # Load directly into df_merged
        logging.info(f"Loaded data from local file: {file2_path}. Shape: {df_merged.shape}")
    except FileNotFoundError:
        logging.error(f"File not found: {file2_path}. Please ensure data2.csv is in the 'data' folder.")
        return pd.DataFrame(), None
    except Exception as e:
        logging.error(f"Error reading local CSV {file2_path}: {e}")
        return pd.DataFrame(), None

    # --- Skip loading df1 and merging --- 

    # --- Feature Engineering & Handling Missing Values --- 
    # Rename columns from data2.csv for consistency
    df_merged.rename(columns={
        'Sleep Duration': 'sleep_duration',
        'Physical Activity Level': 'exercise_freq',
        # Add other renames if needed, e.g., 'Quality of Sleep', 'Stress Level'
    }, inplace=True)
    logging.info("Renamed columns.")

    # Ensure required MODEL_FEATURES exist after renaming
    if 'sleep_duration' not in df_merged.columns:
        logging.error("Column 'sleep_duration' (from 'Sleep Duration') not found after rename. Cannot proceed.")
        return pd.DataFrame(), None
    if 'exercise_freq' not in df_merged.columns:
        logging.error("Column 'exercise_freq' (from 'Physical Activity Level') not found after rename. Imputing randomly.")
        # If critical, return None. If imputation is acceptable:
        df_merged['exercise_freq'] = np.random.randint(0, 7, size=len(df_merged))
    
    # Impute missing values for the model features
    # Sleep Duration
    df_merged['sleep_duration'] = pd.to_numeric(df_merged['sleep_duration'], errors='coerce')
    mean_sleep = df_merged['sleep_duration'].mean()
    df_merged['sleep_duration'].fillna(mean_sleep, inplace=True)
    logging.info(f"Processed 'sleep_duration'. Missing values filled with mean: {mean_sleep:.2f}")

    # Exercise Frequency
    df_merged['exercise_freq'] = pd.to_numeric(df_merged['exercise_freq'], errors='coerce')
    median_exercise = df_merged['exercise_freq'].median()
    df_merged['exercise_freq'].fillna(median_exercise, inplace=True)
    logging.info(f"Processed 'exercise_freq'. Missing values filled with median: {median_exercise}")

    # Process other potentially useful numeric columns (for correlation/summary)
    potential_numeric_cols = ['Age', 'Heart Rate', 'Daily Steps', 'Stress Level', 'Quality of Sleep']
    for col in potential_numeric_cols:
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
            # Optional: Impute NaNs if needed for correlation/summary
            # df_merged[col].fillna(df_merged[col].median(), inplace=True)
        else:
            logging.warning(f"Optional numeric column '{col}' not found.")

    # Encode categorical features 
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    label_encoders = {}
    for col in categorical_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].astype(str).fillna('Unknown') 
            le = LabelEncoder()
            df_merged[col + '_encoded'] = le.fit_transform(df_merged[col])
            label_encoders[col] = le 
            logging.info(f"Label encoded column: {col}")
        else:
            logging.warning(f"Optional categorical column '{col}' not found for encoding.")

    # Drop rows where key model features are still missing after imputation (shouldn't happen with current logic)
    df_merged.dropna(subset=MODEL_FEATURES, inplace=True)

    if df_merged.empty:
        logging.error("Dataframe is empty after cleaning. Cannot train model.")
        return pd.DataFrame(), None

    # Generate the target variable using the processed 'sleep_duration'
    df_merged[TARGET_VARIABLE] = df_merged['sleep_duration'].apply(generate_risk_label)
    logging.info("Generated risk labels.")

    # --- Model Training --- 
    try:
        X = df_merged[MODEL_FEATURES]
        y = df_merged[TARGET_VARIABLE]

        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=300, random_state=42)
        model.fit(X, y)
        logging.info("Logistic Regression model trained successfully.")

        return df_merged, model
    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        return df_merged, None # Return data but no model

# --- Load data and train model on startup ---
df_global, model_global = load_and_prepare_data()

# --- Helper Function ---
def classify_risk_label(label: int) -> str:
    mapping = {0: "Low", 1: "Moderate", 2: "High"}
    return mapping.get(label, "Unknown")

# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    if model_global is None:
        logging.error("Model not available for prediction.")
        abort(503, description="Model is not trained or loaded.")

    if not request.is_json:
        abort(400, description="Request must be JSON.")

    data = request.get_json()
    try:
        sleep_hours = float(data.get('sleepHours', 7.0))
        exercise_freq = int(data.get('exerciseFreq', 3))
    except (TypeError, ValueError):
        abort(400, description="Invalid input types. 'sleepHours' must be float, 'exerciseFreq' must be int.")

    if not (4 <= sleep_hours <= 12 and 0 <= exercise_freq <= 7):
         logging.warning(f"Input values out of typical range: Sleep={sleep_hours}, Exercise={exercise_freq}")
         # Allow prediction but log warning

    try:
        features_input = np.array([[sleep_hours, exercise_freq]])
        pred_label = model_global.predict(features_input)[0]
        pred_proba = model_global.predict_proba(features_input)[0]

        confidence = float(np.max(pred_proba))
        risk_score = int(confidence * 100) # Simple risk score based on confidence
        classification = classify_risk_label(pred_label)

        logging.info(f"Prediction: Input=({sleep_hours}, {exercise_freq}), Class={classification}, Score={risk_score}")
        return jsonify({
            'risk_score': risk_score,
            'classification': classification,
            'confidence': round(confidence, 3)
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        abort(500, description="Internal server error during prediction.")

@app.route('/simulate', methods=['POST'])
def simulate():
    # This endpoint is similar to /predict but handles multiple scenarios
    # For simplicity, it reuses much of the logic.
    # A more robust implementation might optimize batch predictions.
    if model_global is None:
        logging.error("Model not available for simulation.")
        abort(503, description="Model is not trained or loaded.")

    if not request.is_json:
        abort(400, description="Request must be JSON.")

    data = request.get_json()
    scenarios = data.get('scenarios')
    if not isinstance(scenarios, list):
        abort(400, description="'scenarios' must be a list of objects.")

    results = []
    for i, scenario in enumerate(scenarios):
        try:
            sleep_hours = float(scenario.get('sleepHours', 7.0))
            exercise_freq = int(scenario.get('exerciseFreq', 3))

            if not (4 <= sleep_hours <= 12 and 0 <= exercise_freq <= 7):
                 logging.warning(f"Scenario {i}: Input values out of typical range: Sleep={sleep_hours}, Exercise={exercise_freq}")

            features_input = np.array([[sleep_hours, exercise_freq]])
            pred_label = model_global.predict(features_input)[0]
            pred_proba = model_global.predict_proba(features_input)[0]
            confidence = float(np.max(pred_proba))
            risk_score = int(confidence * 100)
            classification = classify_risk_label(pred_label)

            results.append({
                'sleepHours': sleep_hours,
                'exerciseFreq': exercise_freq,
                'risk_score': risk_score,
                'classification': classification,
                'confidence': round(confidence, 3)
            })
        except (TypeError, ValueError):
            logging.warning(f"Scenario {i}: Invalid input types - skipping.")
            continue # Skip this scenario
        except Exception as e:
            logging.error(f"Error processing scenario {i}: {e}")
            # Optionally append an error object to results
            continue

    logging.info(f"Simulation complete. Processed {len(results)} out of {len(scenarios)} scenarios.")
    return jsonify(results)

@app.route('/summary', methods=['GET'])
def summary():
    if df_global is None or df_global.empty:
        logging.warning("Summary requested, but data is not available.")
        abort(404, description="Data not available for summary.")

    try:
        # Calculate summary statistics, handling potential missing columns
        summary_stats = {}
        if 'sleep_duration' in df_global.columns:
            summary_stats["sleep_duration"] = {
                "mean": round(df_global['sleep_duration'].mean(), 2),
                "median": round(df_global['sleep_duration'].median(), 2),
                "min": round(df_global['sleep_duration'].min(), 2),
                "max": round(df_global['sleep_duration'].max(), 2)
            }
        if 'exercise_freq' in df_global.columns:
             summary_stats["exercise_freq"] = {
                "mean": round(df_global['exercise_freq'].mean(), 2),
                "median": round(df_global['exercise_freq'].median(), 2),
                "min": round(df_global['exercise_freq'].min(), 2),
                "max": round(df_global['exercise_freq'].max(), 2)
            }
        if TARGET_VARIABLE in df_global.columns:
            # Convert numeric labels back to strings for readability
            risk_counts = df_global[TARGET_VARIABLE].map(classify_risk_label).value_counts().to_dict()
            summary_stats["risk_distribution"] = risk_counts

        logging.info("Generated data summary.")
        return jsonify(summary_stats)
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        abort(500, description="Internal server error during summary generation.")

@app.route('/correlation', methods=['GET'])
def correlation():
    if df_global is None or df_global.empty:
        logging.warning("Correlation requested, but data is not available.")
        abort(404, description="Data not available for correlation analysis.")

    try:
        # Select only numeric columns for correlation matrix
        numeric_df = df_global.select_dtypes(include=np.number)
        numeric_cols = numeric_df.columns.tolist()
        logging.info(f"Numeric columns identified for correlation: {numeric_cols}")

        # Optional: Exclude less relevant numeric columns if needed
        # Consider excluding encoded categoricals unless desired
        cols_to_exclude = [col for col in numeric_cols if '_encoded' in col] # Exclude label encoded cols by default
        # Add other IDs if necessary: e.g., cols_to_exclude.extend(['UserID', 'Person ID'])
        
        # Filter out explicitly excluded columns
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
        logging.info(f"Numeric columns after exclusion: {numeric_cols}")

        if not numeric_cols:
            logging.error("No suitable numeric columns found after exclusion for correlation analysis.")
            abort(400, description="No numeric columns found for correlation analysis.")

        # Calculate correlation on the filtered numeric columns
        correlation_matrix = df_global[numeric_cols].corr()
        logging.info(f"Calculated correlation matrix (shape: {correlation_matrix.shape})")

        # Check for issues in the matrix (e.g., all NaN)
        if correlation_matrix.isnull().all().all():
             logging.warning("Correlation matrix contains all NaN values.")
             # Decide how to handle: return empty, error, or the NaN matrix? Returning error for now.
             abort(500, description="Could not calculate valid correlations.")

        # Convert matrix to JSON-friendly format (list of dictionaries)
        # Fill NaN with null or a placeholder string for JSON compatibility
        correlation_data = correlation_matrix.fillna('N/A').reset_index().to_dict(orient='records')
        logging.info("Formatted correlation matrix for JSON response.")

        # Also send column names for heatmap axes
        response_data = {
            'columns': numeric_cols, # Send the columns actually used
            'correlation': correlation_data
        }

        logging.info("Generated correlation matrix successfully.")
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Error generating correlation matrix: {e}", exc_info=True) # Log traceback
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
    # Use environment variable for port or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Set debug=False for production
    app.run(host='0.0.0.0', port=port, debug=True)
