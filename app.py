from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
from src.feature_store import FeatureStore

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "artifacts/models/random_forest_model.pkl"

def load_model():
    """Load the trained model"""
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
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
    
    # Create feature vector in the correct order
    features = [
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
    
    return np.array(features).reshape(1, -1)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input data"""
    try:
        if model is None:
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
        
        # Preprocess the input
        features = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
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
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)