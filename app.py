from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load the trained model and preprocessing components
model = joblib.load('rain_prediction_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Initialize the Flask application
app = Flask(__name__)

def preprocess_input(data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])
    
    # Normalize categorical fields to lowercase before label encoding
    categorical_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'Location']
    
    for col in categorical_columns:
        if col in input_df.columns:
            input_df[col] = input_df[col].str.lower()  # Convert to lowercase
            input_df[col] = label_encoders[col].transform(input_df[col])
    
    # Scale numerical features
    num_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 
                'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 
                'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 
                'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 
                'Temp9am', 'Temp3pm']
    
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    return input_df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from request
    processed_data = preprocess_input(data)  # Preprocess the input data
    
    # Make prediction using the model
    prediction = model.predict(processed_data)
    prediction_probabilities = model.predict_proba(processed_data)  # Get probabilities
    
    # Extract the probability for the "Yes" (rain tomorrow) class, which is typically index 1
    probability_yes = prediction_probabilities[0][1]
    probability_no = prediction_probabilities[0][0]  # Probability of 'No'
    
    # Convert prediction to human-readable format
    result = "Yes" if prediction[0] == 1 else "No"
    
    # Print for debugging purposes
    print(f"Prediction: {result}, Probability of Yes: {probability_yes}, Probability of No: {probability_no}")
    
    return jsonify({
        "RainTomorrow": result,
        "Probability_Yes": probability_yes,
        "Probability_No": probability_no  # Return both probabilities
    })


if __name__ == '__main__':
    app.run(debug=True)
