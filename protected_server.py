import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
import os
from peewee import (SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError, CharField)

from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

########################################
# Database configuration

#DB = SqliteDatabase('predictions.db')
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = CharField(primary_key=True, max_length=50)
    prediction = IntegerField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

########################################
# Load the trained model

with open('columns.json') as fh:
    columns = json.load(fh)

with open('pipeline.pickle', 'rb') as fh:
    pipeline = joblib.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# End model loading
########################################

########################################
# Input validation functions

def extract_model_features(input_data):
    """
    Extracts the necessary features for the model from the full input.
    
    Parameters:
    input_data (dict): The full JSON input received in the request
    
    Returns:
    dict: A dictionary containing only the features needed for the model
    """
    # Define the columns the model expects
    required_columns = {
        "Facility Id",
        "Age Group",
        "Race",
        "Ethnicity",
        "Type of Admission",
        "CCS Diagnosis Code",
        "CCS Procedure Code",
        "APR DRG Code",
        "APR MDC Code",
        "APR Severity of Illness Code",
        "APR Risk of Mortality",
        "Emergency Department Indicator",
        "Gender",
        'Health Service Area',
        'Attending Provider License Number'
    }
    
    # Numeric columns that need conversion
    numeric_columns = {
        "Facility Id", 
        "CCS Diagnosis Code",
        "CCS Procedure Code",
        "APR DRG Code",
        "APR MDC Code",
        "APR Severity of Illness Code",
        'Attending Provider License Number'
    }
    # Remove observation_id from input to not consider it as a feature
    features_data = {k: v for k, v in input_data.items() if k != 'observation_id'}
    
    # Extract only the necessary columns
    model_features = {}
    missing_columns = []
    
    for column in required_columns:
        if column in features_data:
            value = features_data[column]
            
            # Check if the value is empty or None
            if value is None or value == '':
                missing_columns.append(column)
            else:
                # Convert strings to numbers where appropriate
                if column in numeric_columns:
                    if isinstance(value, str):
                        if value.lower() == 'nan':
                            value = 0  # or another appropriate default value
                        elif value.replace('.', '', 1).isdigit():
                            value = float(value) if '.' in value else int(value)
                
                model_features[column] = value
        else:
            missing_columns.append(column)
    
    # If any required columns are missing or empty, raise an error
    if missing_columns:
        error_msg = f"All fields are mandatory. Missing or empty values for: {', '.join(missing_columns)}"
        raise ValueError(error_msg)
    
    return model_features



def validate_input_values(features):
    """
    Validates the categorical and numeric values in the extracted features.
    
    Parameters:
    features (dict): Dictionary containing the extracted features for the model
    
    Returns:
    tuple: (is_valid, error_message)
    """
    # Validate categorical values
    valid_category_map = {
        "Age Group": ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"],
        "Race": ["White", "Other Race", "Black/African American", "Multi-racial"],
        "Ethnicity": ["Not Span/Hispanic", "Spanish/Hispanic", "Unknown", "Multi-ethnic"],
        "Type of Admission": ["Emergency", "Elective", "Newborn", "Urgent", "Trauma", "Not Available"],
        "APR Risk of Mortality": ["Minor", "Moderate", "Major", "Extreme"],
        "Emergency Department Indicator": ["Y", "N"],
        "Gender": ["M", "F"],
        'Health Service Area':["New York City","Long Island","Hudson Valley","Capital/Adiron",
                                "Western NY", "Central NY","Finger Lakes","Southern Tier"],
    }
    
    for field, valid_values in valid_category_map.items():
        if features[field] not in valid_values:
            return False, f"Invalid value for {field}: '{features[field]}'. Allowed values: {', '.join(valid_values)}"
    
    # Validate numeric values
    valid_range_map = {
        "Facility Id": (1, 9431), 
        "CCS Diagnosis Code": (1, 917),
        "CCS Procedure Code": (0, 999),
        "APR DRG Code": (1, 956),
        "APR MDC Code": (0, 25),
        "APR Severity of Illness Code": (0, 4),
        'Attending Provider License Number': (0, 90999999)
    }
    
    for field, (min_val, max_val) in valid_range_map.items():
        value = features[field]
        if not isinstance(value, (int, float)):
            return False, f"Invalid type for {field}: {type(value).__name__}. Expected a number."
        if not (min_val <= value <= max_val):
            return False, f"Invalid value for {field}: {value}. Should be between {min_val} and {max_val}."
    
    return True, ""

########################################
# Begin web server setup

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the full input data
    input_data = request.get_json()
    
    # Check if observation_id exists
    #if "observation_id" not in input_data:
    if "observation_id" not in input_data or not isinstance(input_data["observation_id"], str) or input_data["observation_id"] == "":
        return jsonify({'error': "Field 'observation_id' not found in input"})
    
    observation_id = input_data['observation_id']
    
    try:
        # Extract only the features necessary for the model from the full input
        model_features = extract_model_features(input_data)
        
        # Validate the extracted features
        valid, error_msg = validate_input_values(model_features)
        if not valid:
            return jsonify({'error': error_msg})
        
        # Create a DataFrame only with the model features
        features_df = pd.DataFrame([model_features], columns=columns).astype(dtypes)
        
        # Generate prediction
        prediction = pipeline.predict(features_df)[0]
        
        # Create response
        response = {
            'observation_id': observation_id,
            'prediction': int(prediction),
            'features_used': list(model_features.keys())
        }
        
        # Save to the database
        try:
            Prediction.create(
                observation_id=observation_id,
                observation=json.dumps(input_data),  # Save the original full input
                prediction=int(prediction)
            )
        except IntegrityError:
            response['warning'] = f"Observation ID '{observation_id}' already exists. Not saved to database."
            DB.rollback()
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': str(e)})
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {str(e)}"})

@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        return jsonify({'error': f'Observation ID "{obs["observation_id"]}" does not exist'})

#
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5010)