import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

########################################
# Configuração do banco de dados

#DB = SqliteDatabase('predictions.db')
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = CharField(primary_key=True, max_length=50)
    observation = TextField()
    prediction = IntegerField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

########################################
# Carregamento do modelo treinado

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

def check_request(request):
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if the request is okay, False otherwise
        - error message: empty if the request is okay, error message otherwise
    """
    
    if "observation_id" not in request:
        error = "Field `observation_id` missing from request: {}".format(request)
        return False, error
    
    if "observation" not in request:
        error = "Field `observation` missing from request: {}".format(request)
        return False, error
    
    return True, ""


def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, error message otherwise
    """
    
    valid_columns = {
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
        "APR Risk of Mortality"
    }
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""


def check_categorical_values(observation):
    """
        Validates that all categorical fields in the observation are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, error message otherwise
    """
    
    valid_category_map = {
        "Age Group": ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"],
        "Race": ["White", "Other Race", "Black/African American", "Multi-racial"],
        "Ethnicity": ["Not Span/Hispanic", "Spanish/Hispanic", "Unknown", "Multi-ethnic"],
        "Type of Admission": ["Emergency", "Elective", "Newborn", "Urgent", "Trauma", "Not Available"],
        "APR Risk of Mortality": ["Minor", "Moderate", "Major", "Extreme"],
    }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error

    return True, ""


def check_numerical_values(observation):
    """
    Validates that all numerical fields in the observation are valid.

    Returns:
    - assertion value: True if all provided numerical fields have valid values,
      False otherwise.
    - error message: empty if all fields are valid, an error message if any field is invalid.
    """
    
    valid_numerical_map = {
        "Facility Id": (1, 9431), 
        "CCS Diagnosis Code": (1, 917),
        "CCS Procedure Code": (0, 999),
        "APR DRG Code": (1, 956),
        "APR MDC Code": (0, 25),
        "APR Severity of Illness Code": (0, 4),
    }
    
    for key, (min_val, max_val) in valid_numerical_map.items():
        if key in observation:
            value = observation[key]
            # Check if the value is numeric
            if not isinstance(value, (int, float)):
                error = "Invalid type for {}: {}. Expected a number.".format(key, type(value).__name__)
                return False, error
            # Check if the value is within the expected range
            if not (min_val <= value <= max_val):
                error = "Invalid value for {}: {}. Expected a value between {} and {}.".format(
                    key, value, min_val, max_val)
                return False, error
        else:
            error = "Numerical field {} missing.".format(key)
            return False, error

    return True, ""


def filter_valid_columns(observation):
    """
    Filters the observation to keep only the valid columns (the ones the model expects).
    
    Returns:
    - filtered observation: a dictionary with only the valid columns
    """
    valid_columns = {
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
        "APR Risk of Mortality"
    }
    
    # Keep only the valid columns in the observation
    filtered_observation = {key: value for key, value in observation.items() if key in valid_columns}
    
    return filtered_observation


# End input validation functions
########################################

########################################
# Begin web server setup

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()

    request_ok, error = check_request(obs_dict)
    if not request_ok:
        return jsonify({'error': error})

    _id = obs_dict['observation_id']
    observation = filter_valid_columns(obs_dict['observation'])

    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        return jsonify({'error': error})

    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    prediction = pipeline.predict(obs)[0]

    response = {'observation_id': _id, 'prediction': prediction}

    try:
        Prediction.create(
            observation_id=_id,
            observation=json.dumps(obs_dict['observation']),
            prediction=prediction
        )
    except IntegrityError:
        response['error'] = f"Observation ID '{_id}' already exists."
        DB.rollback()
    
    return jsonify(response)

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5010)

