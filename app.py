import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        input_values = [float(request.form[field_name]) for field_name in request.form]
        np_data = np.asarray(input_values, dtype=np.float32)
        
        # Make a prediction
        prediction = model.predict(np_data.reshape(1, -1))
        
        # Generate output message
        if prediction[0] == 1:
            output = "This person has Parkinson's disease."
        else:
            output = "This person does not have Parkinson's disease."
    except ValueError:
        output = "Invalid input. Please enter valid numerical values for all fields."
    except Exception as e:
        output = f"An error occurred: {str(e)}"

    return render_template("index.html", message=output)

if __name__ == "__main__":
    app.run(debug=True)
