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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define features the model will now use
# Note: Gender will be encoded, so we use 'Gender_encoded'
MODEL_FEATURES = ['sleep_duration', 'exercise_freq', 'Age', 'Gender_encoded', 'Stress Level']
TARGET_VARIABLE = 'risk_label'

# Global variables to store model, data summary, and accuracy
df_summary_global = None # Store processed data for summary/correlation
pipeline_global = None # Store the trained pipeline (includes scaling + model)
model_accuracy_global = None # Store accuracy
feature_names_global = None # Store feature names used by the final model

# --- Helper Functions (keep generate_risk_label, classify_risk_label) ---
def generate_risk_label(sleep_duration):
    # 0: Low (>= 8 hrs), 1: Moderate (6-8 hrs), 2: High (< 6 hrs)
    if pd.isna(sleep_duration): return 1 # Moderate if unknown
    if sleep_duration < 6: return 2      # High
    elif sleep_duration < 8: return 1   # Moderate (Changed from <= 8)
    else: return 0                      # Low

def classify_risk_label(label: int) -> str:
    mapping = {0: "Low", 1: "Moderate", 2: "High"}
    return mapping.get(label, "Unknown")

# --- Data Loading and Preprocessing ---
def load_and_prepare_data():
    global feature_names_global # To store final feature names
    logging.info("Loading and preparing data from local file data/data2.csv...")

    data_dir = 'data'
    file_path = os.path.join(data_dir, 'data2.csv')

    # Ensure data directory exists (moved check here)
    if not os.path.exists(data_dir):
        logging.error(f"Data directory '{data_dir}' not found. Please create it.")
        return None, None, None 
    # Ensure file exists
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}. Please ensure data2.csv is in the '{data_dir}' folder.")
        return None, None, None 

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from local file: {file_path}. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error reading local CSV {file_path}: {e}")
        return None, None, None

    # --- Feature Engineering & Handling Missing Values ---
    df.rename(columns={
        'Sleep Duration': 'sleep_duration',
        'Physical Activity Level': 'exercise_freq',
        # Keep original 'Age', 'Gender', 'Stress Level'
    }, inplace=True)
    logging.info("Renamed columns.")

    # Convert key columns to numeric, coercing errors
    numeric_cols_to_process = ['sleep_duration', 'exercise_freq', 'Age', 'Stress Level', 'Heart Rate', 'Daily Steps', 'Quality of Sleep']
    for col in numeric_cols_to_process:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logging.warning(f"Column '{col}' not found. It might be needed later.")

    # Impute missing numeric values (using median for robustness)
    cols_to_impute = ['sleep_duration', 'exercise_freq', 'Age', 'Stress Level']
    for col in cols_to_impute:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logging.info(f"Imputed missing values in '{col}' with median: {median_val}")
        else:
            # Handle critical missing columns needed for model
             if col in ['sleep_duration', 'exercise_freq', 'Age', 'Stress Level']:
                 logging.error(f"Critical model feature column '{col}' is missing after rename. Cannot proceed.")
                 return None, None, None

    # Encode Gender
    if 'Gender' in df.columns:
        le = LabelEncoder()
        df['Gender_encoded'] = le.fit_transform(df['Gender'].astype(str).fillna('Unknown'))
        logging.info("Label encoded column: Gender")
        # Store mapping if needed: gender_map = dict(zip(le.classes_, le.transform(le.classes_)))
    else:
        logging.error("Column 'Gender' is missing. Cannot create 'Gender_encoded' feature.")
        return None, None, None

    # Define columns for preprocessing pipeline
    numeric_features = ['sleep_duration', 'exercise_freq', 'Age', 'Stress Level']
    # Categorical features are already handled ('Gender_encoded')
    
    # Check if all required MODEL_FEATURES are now present
    missing_features = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing_features:
         logging.error(f"Required model features missing before training: {missing_features}. Cannot train model.")
         return None, None, None

    # Generate the target variable
    df[TARGET_VARIABLE] = df['sleep_duration'].apply(generate_risk_label)
    logging.info("Generated risk labels.")

    # Drop rows where target is somehow NaN (shouldn't happen now)
    df.dropna(subset=[TARGET_VARIABLE], inplace=True)
    
    # --- Model Training with Pipeline and Train/Test Split ---
    try:
        X = df[MODEL_FEATURES]
        y = df[TARGET_VARIABLE]

        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        logging.info(f"Data split: Train shape {X_train.shape}, Test shape {X_test.shape}")

        # Create preprocessing pipeline (only scaling numeric features here)
        # 'passthrough' could be used for 'Gender_encoded' if no scaling needed
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features) 
                # Add ('cat', OneHotEncoder(), categorical_features) if using OHE
            ],
            remainder='passthrough' # Keep 'Gender_encoded' as is
        )
        
        # Create full pipeline including model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression(multi_class='ovr', # ovr often works well
                                                                solver='liblinear', # Good solver for ovr
                                                                random_state=42,
                                                                class_weight='balanced')) # Handle imbalanced classes
                               ])

        # Train the pipeline
        pipeline.fit(X_train, y_train)
        logging.info("Preprocessing + Logistic Regression pipeline trained successfully.")

        # Evaluate model accuracy on test set
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Pipeline accuracy on test set: {accuracy:.3f}")

        # Get feature names after preprocessing (important for interpretation if needed)
        # Accessing feature names from ColumnTransformer can be complex, storing original for now
        feature_names_global = MODEL_FEATURES 

        return df, pipeline, accuracy # Return full df, trained pipeline, and accuracy

    except Exception as e:
        logging.error(f"Error during model training pipeline: {e}", exc_info=True)
        return df, None, None # Return data but no pipeline/accuracy

# --- Load data and train model on startup ---
df_summary_global, pipeline_global, model_accuracy_global = load_and_prepare_data()

# --- API Endpoints ---

# NEW Endpoint for Model Info
@app.route('/model-info', methods=['GET'])
def model_info():
     if model_accuracy_global is None:
          # Return an error or default info if accuracy wasn't calculated
          return jsonify({"error": "Model accuracy not available."}), 503
     else:
          return jsonify({
               "accuracy": round(model_accuracy_global, 3),
               "features_used": feature_names_global,
               "target": TARGET_VARIABLE,
               "model_type": "Logistic Regression (in Pipeline)"
          })

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline_global is None: # Check pipeline now
        logging.error("Pipeline not available for prediction.")
        abort(503, description="Model Pipeline is not trained or loaded.")

    if not request.is_json: abort(400, description="Request must be JSON.")

    data = request.get_json()
    is_simulation = 'age' not in data # Heuristic: Assume simulation if 'age' is missing

    try:
        # Extract all expected features, providing defaults for simulation
        sleep_hours = float(data.get('sleepHours')) # Required
        exercise_freq = int(data.get('exerciseFreq')) # Required
        age = int(data.get('age', 35)) # Default age if missing
        gender = str(data.get('gender', 'Female')) # Default gender if missing
        stress_level = int(data.get('stressLevel', 5)) # Default stress level if missing

        # Basic validation (allow defaults to pass)
        if not all(isinstance(x, (int, float)) for x in [sleep_hours, exercise_freq, age, stress_level]):
             raise ValueError("Numeric inputs required.")
        if gender not in ['Male', 'Female', 'Other']: # Allow 'Other' or map it
             # If LabelEncoder only saw Male/Female, 'Other' might cause issues
             # For now, let's default 'Other' to 'Female' encoding during prediction
             logging.warning(f"Received gender '{gender}', mapping to 'Female' for prediction.")
             gender = 'Female'
             # Alternatively, handle 'Other' in encoding/training if possible

    except (TypeError, ValueError, KeyError) as e:
        # KeyErrors should be less likely now with .get defaults, but keep for safety
        logging.error(f"Invalid input data: {e} - Data received: {data}")
        abort(400, description=f"Invalid input data format or values: {e}")

    try:
        # Manually encode gender based on training
        # IMPORTANT: This assumes the LabelEncoder used during training
        # fit the data such that Female maps to 0 and Male maps to 1.
        # If the training changes or includes other genders, this MUST be updated
        # or the encoding step moved into the scikit-learn pipeline itself.
        gender_encoded = 1 if gender == 'Male' else 0

        # Create DataFrame for prediction with correct feature names IN ORDER
        input_df = pd.DataFrame([[\
            sleep_hours, exercise_freq, age, gender_encoded, stress_level\
        ]], columns=MODEL_FEATURES) # Use the exact feature list

        # Log differently for simulation vs user input
        log_prefix = "Simulation Prediction" if is_simulation else "User Prediction"
        # Use debug level for potentially sensitive or verbose data
        logging.debug(f"{log_prefix} - Input DataFrame:\\\\n{input_df}")

        # Use pipeline to predict (handles preprocessing + prediction)
        pred_label = pipeline_global.predict(input_df)[0]
        pred_proba = pipeline_global.predict_proba(input_df)[0]

        # --- MODIFIED Risk Score Calculation ---
        # Use probability of the highest risk class (label 2)
        # Ensure the index 2 correctly corresponds to the 'High' risk label mapping
        high_risk_proba = pred_proba[2] if len(pred_proba) > 2 else 0 
        risk_score = int(high_risk_proba * 100) 
        # Alternative: Map label to score: risk_map = {0: 25, 1: 50, 2: 75}; risk_score = risk_map.get(pred_label, 50)

        classification = classify_risk_label(pred_label)
        # Confidence remains the max probability, indicating model's certainty in its chosen class
        confidence = float(np.max(pred_proba))

        logging.info(f"{log_prefix}: Input=(sleep={sleep_hours}, exercise={exercise_freq}, age={age}, gender={gender}, stress={stress_level}), Label={pred_label}, Class={classification}, Score={risk_score}, Confidence={confidence:.3f}")
        return jsonify({
            'risk_score': risk_score, # Now based on high-risk probability
            'classification': classification,
            'confidence': round(confidence, 3)
        })
    except IndexError:
        logging.error(f"Error accessing predicted probabilities. Probabilities array: {pred_proba}. Check model output.", exc_info=True)
        abort(500, description="Internal server error: Could not determine risk probability.")
    except Exception as e:
        logging.error(f"Error during prediction pipeline: {e}", exc_info=True)
        abort(500, description="Internal server error during prediction.")


# --- Update /simulate similarly if needed ---
# @app.route('/simulate', methods=['POST']) ... update to handle new features ...


@app.route('/summary', methods=['GET'])
def summary():
    if df_summary_global is None or df_summary_global.empty:
        logging.warning("Summary requested, but data is not available.")
        abort(404, description="Data not available for summary.")
    # (Keep existing summary logic, it uses df_summary_global implicitly)
    # ... (rest of summary function) ...
    try:
        summary_stats = {}
        # Example: Use original column names before rename if preferred for display
        if 'sleep_duration' in df_summary_global.columns:
            # Calculate stats on the processed column
             summary_stats["Sleep Duration (hours)"] = { # User-friendly name
                "mean": round(df_summary_global['sleep_duration'].mean(), 2),
                "median": round(df_summary_global['sleep_duration'].median(), 2),
                "min": round(df_summary_global['sleep_duration'].min(), 2),
                "max": round(df_summary_global['sleep_duration'].max(), 2)
            }
        # ... repeat for exercise_freq, Age, Stress Level etc. ...
        if 'exercise_freq' in df_summary_global.columns:
             summary_stats["Exercise Frequency (days/wk)"] = {
                 "mean": round(df_summary_global['exercise_freq'].mean(), 2),
                 "median": round(df_summary_global['exercise_freq'].median(), 2),
                 "min": round(df_summary_global['exercise_freq'].min(), 2),
                 "max": round(df_summary_global['exercise_freq'].max(), 2)
             }
        if TARGET_VARIABLE in df_summary_global.columns:
            risk_counts = df_summary_global[TARGET_VARIABLE].map(classify_risk_label).value_counts().to_dict()
            summary_stats["Predicted Risk Distribution"] = risk_counts # Clearer title

        logging.info("Generated data summary.")
        return jsonify(summary_stats)
    except Exception as e:
        logging.error(f"Error generating summary: {e}", exc_info=True)
        abort(500, description="Internal server error during summary generation.")


@app.route('/correlation', methods=['GET'])
def correlation():
    if df_summary_global is None or df_summary_global.empty:
        logging.warning("Correlation requested, but data is not available.")
        abort(404, description="Data not available for correlation analysis.")
    # (Keep existing correlation logic, uses df_summary_global implicitly)
    # ... (rest of correlation function) ...
    try:
        numeric_df = df_summary_global.select_dtypes(include=np.number)
        numeric_cols = numeric_df.columns.tolist()
        logging.info(f"Numeric columns identified for correlation: {numeric_cols}")

        # Exclude encoded categoricals and IDs
        cols_to_exclude = [col for col in numeric_cols if '_encoded' in col or 'ID' in col or 'Id' in col]
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
        logging.info(f"Numeric columns after exclusion: {numeric_cols}")

        if not numeric_cols:
            logging.error("No suitable numeric columns found after exclusion for correlation analysis.")
            abort(400, description="No numeric columns found for correlation analysis.")

        correlation_matrix = df_summary_global[numeric_cols].corr()
        logging.info(f"Calculated correlation matrix (shape: {correlation_matrix.shape})")

        if correlation_matrix.isnull().all().all():
             logging.warning("Correlation matrix contains all NaN values.")
             abort(500, description="Could not calculate valid correlations.")

        correlation_data = correlation_matrix.fillna('N/A').reset_index().to_dict(orient='records')
        logging.info("Formatted correlation matrix for JSON response.")

        response_data = {'columns': numeric_cols, 'correlation': correlation_data }
        logging.info("Generated correlation matrix successfully.")
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Error generating correlation matrix: {e}", exc_info=True)
        abort(500, description="Internal server error during correlation analysis.")


# --- Error Handlers (keep existing) ---
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

# --- Main Execution (keep existing) ---
if __name__ == '__main__':
    # Use environment variable for port or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Set debug=False for production
    app.run(host='0.0.0.0', port=port, debug=True)
