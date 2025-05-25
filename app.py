from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
from alibi_detect.cd import KSDrift
from sklearn.preprocessing import StandardScaler
from src.feature_store import FeatureStore
from src.logger import get_logger
from prometheus_client import start_http_server, Counter, Gauge

logger = get_logger(__name__)

app = Flask(__name__)

prediction_count = Counter('prediction_count', 'Number of predictions made')
drift_count = Counter('drift_count', 'Number of drift detections')

# Load the trained model
MODEL_PATH = "artifacts/models/random_forest_model.pkl"

def load_model():
    """Load the trained model"""
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

model = load_model()

def extract_title(name):
    """Extract title from passenger name"""
    if pd.isna(name):
        return 'Unknown'
    
    title_mapping = {
        'Mr.': 'Mr',
        'Mrs.': 'Mrs',
        'Miss.': 'Miss',
        'Master.': 'Master',
        'Dr.': 'Dr',
        'Rev.': 'Rev',
        'Col.': 'Col',
        'Major.': 'Major',
        'Mlle.': 'Miss',
        'Countess.': 'Mrs',
        'Ms.': 'Miss',
        'Lady.': 'Mrs',
        'Jonkheer.': 'Mr',
        'Don.': 'Mr',
        'Dona.': 'Mrs',
        'Mme.': 'Mrs',
        'Capt.': 'Col',
        'Sir.': 'Mr'
    }
    
    for title_key, title_value in title_mapping.items():
        if title_key in name:
            return title_value
    return 'Other'

def preprocess_input(data):
    """Preprocess input data to match training format"""
    # Calculate derived features
    data['Familysize'] = data['SibSp'] + data['Parch'] + 1
    data['Isalone'] = 1 if data['Familysize'] == 1 else 0
    data['HasCabin'] = 1 if data.get('Cabin', '') else 0
    data['Title'] = extract_title(data.get('Name', ''))
    data['Pclass_Fare'] = data['Pclass'] * data['Fare']
    data['Age_Fare'] = data['Age'] * data['Fare']
    
    # Encode categorical variables
    sex_encoded = 1 if data['Sex'] == 'male' else 0
    embarked_encoded = {'S': 0, 'C': 1, 'Q': 2}.get(data['Embarked'], 0)
    title_encoded = {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Dr': 4, 'Rev': 5, 'Col': 6, 'Other': 7}.get(data['Title'], 7)
    
    # Create feature vector with proper feature names
    feature_values = [
        data['Age'],
        data['Fare'],
        data['Pclass'],
        sex_encoded,
        embarked_encoded,
        data['Familysize'],
        data['Isalone'],
        data['HasCabin'],
        title_encoded,
        data['Pclass_Fare'],
        data['Age_Fare']
    ]
    
    # Return as DataFrame with feature names to avoid sklearn warnings
    feature_names = ['Age', 'Fare', 'Pclass', 'Sex', 'Embarked', 'Familysize', 'Isalone', 'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']
    return pd.DataFrame([feature_values], columns=feature_names)

feature_store = FeatureStore()
# Use actual feature names as stored in Redis
features = ['Age', 'Fare', 'Pclass', 'Sex', 'Embarked', 'Familysize', 'Isalone', 'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']
scaler = StandardScaler()

def fit_scaler_on_ref_data():
    entity_ids = feature_store.get_all_entity_ids()
    if not entity_ids:
        logger.warning("No entity IDs found in Redis. Please run the training pipeline first.")
        return None
    
    all_features = feature_store.get_batch_features(entity_ids)
    # Filter out None values and create DataFrame
    valid_features = {k: v for k, v in all_features.items() if v is not None}
    
    if not valid_features:
        logger.warning("No valid features found in Redis.")
        return None
        
    all_features_df = pd.DataFrame.from_dict(valid_features, orient='index')[features]
    scaler.fit(all_features_df)
    logger.info(f"Scaler fitted on {len(valid_features)} reference data points")
    return all_features_df.values

# Initialize historical data and drift detector only if data is available
historical_data = fit_scaler_on_ref_data()
if historical_data is not None:
    ksd = KSDrift(x_ref=historical_data, p_val=0.05)
    logger.info("Drift detector initialized successfully")
else:
    ksd = None
    logger.warning("Warning: Drift detection not available. Run training pipeline first.")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input data"""
    try:
        if model is None:
            logger.error("Model not loaded for prediction")
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get form data
        data = {
            'Age': float(request.form.get('age', 0)),
            'Fare': float(request.form.get('fare', 0)),
            'Pclass': int(request.form.get('pclass', 3)),
            'Sex': request.form.get('sex', 'male'),
            'Embarked': request.form.get('embarked', 'S'),
            'SibSp': int(request.form.get('sibsp', 0)),
            'Parch': int(request.form.get('parch', 0)),
            'Name': request.form.get('name', ''),
            'Cabin': request.form.get('cabin', '')
        }
        
        logger.info(f"Prediction request for passenger: {data.get('Name', 'Anonymous')}")
        
        # Preprocess the input (now returns DataFrame)
        features_df = preprocess_input(data)
        
        # Check for drift only if drift detector is available
        if ksd is not None and historical_data is not None:
            features_scaled = scaler.transform(features_df)
            drift = ksd.predict(features_scaled)
            if drift['data']['is_drift']:
                logger.warning(f"Data drift detected for prediction")
                drift_count.inc()
            else:
                logger.info("No data drift detected")
        else:
            logger.debug("Drift detection skipped - no reference data available")
        
        # Make prediction using DataFrame
        prediction = model.predict(features_df)[0]
        prediction_count.inc()

        probability = model.predict_proba(features_df)[0]
        
        logger.info(f"Prediction completed: survived={bool(prediction)}, probability={float(probability[1]):.3f}")
        
        # Prepare response
        result = {
            'survived': bool(prediction),
            'survival_probability': float(probability[1]),
            'death_probability': float(probability[0]),
            'passenger_info': {
                'name': data['Name'] or 'Anonymous Passenger',
                'age': data['Age'],
                'sex': data['Sex'],
                'pclass': data['Pclass'],
                'fare': data['Fare'],
                'embarked': data['Embarked']
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics"""
    from prometheus_client import generate_latest
    from flask import Response
    return Response(generate_latest(), content_type='text/plain')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    start_http_server(port=8000)
    logger.info("Starting Flask application on host=0.0.0.0, port=5000")
    app.run(debug=False, host='0.0.0.0', port=5000)