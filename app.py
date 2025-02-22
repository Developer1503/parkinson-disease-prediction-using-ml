from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Gradient Boosting model
model = pickle.load(open('gb_model.pkl', 'rb'))

# Load the scaler used during training
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        input_values = [float(request.form[field_name]) for field_name in request.form]
        np_data = np.asarray(input_values, dtype=np.float32).reshape(1, -1)

        # Scale the input data
        np_data_scaled = scaler.transform(np_data)

        # Make a prediction
        prediction = model.predict(np_data_scaled)
        prediction_proba = model.predict_proba(np_data_scaled)

        # Generate output message
        if prediction[0] == 1:
            output = "This person has Parkinson's disease."
        else:
            output = "This person does not have Parkinson's disease."

        confidence = f"Confidence: {prediction_proba[0][prediction[0]]:.2f}"
        result = f"{output} {confidence}"

        return result
    except ValueError:
        return "Invalid input. Please enter valid numerical values for all fields."
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
