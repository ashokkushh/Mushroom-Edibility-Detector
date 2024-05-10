import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from flask import jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [request.form.values()]
    columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
               'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 
               'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
               'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 
               'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
    input_df = pd.DataFrame(input_data, columns=columns)
    input_features_encoded = pd.get_dummies(input_df)
    
    # Align columns with the model's feature names
    missing_cols = set(model.feature_names_in_) - set(input_features_encoded.columns)
    for col in missing_cols:
        input_features_encoded[col] = 0
    input_features_encoded = input_features_encoded[model.feature_names_in_]

    # Make prediction
    prediction = model.predict(input_features_encoded)
    if prediction == 'e':
        result = 'Edible'
    else:
        result = 'Poisonous'

    # Render the home page with prediction result
    #return render_template('index.html', prediction_text='The Mushroom is {}'.format(result))
    return jsonify(prediction_text='The Mushroom is {}'.format(result))
if __name__ == "__main__":
    app.run(debug= False, host='0.0.0.0')
