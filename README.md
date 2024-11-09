Parkinson's Disease Prediction Web Application
This project is a web application that predicts the likelihood of a person having Parkinson's disease based on certain medical voice measurements. It uses a trained machine learning model (SVM and Decision Tree) to classify the data.

Features
Flask Web Application: Users can input medical data through a simple web interface.
Machine Learning Model: Supports Parkinson’s prediction using SVM and Decision Tree models.
Data Standardization and Processing: Ensures accurate predictions by scaling and transforming input data.
Real-time Prediction Results: Shows results immediately upon input.
Dataset
The dataset used for training is a Parkinson’s disease dataset containing various medical measurements.

Project Structure
app.py: Contains the Flask application and routes for the web interface.
model.pkl: Pickled file of the trained machine learning model.
templates/index.html: HTML template for the web interface.
Dependencies
Flask
sklearn
pandas
numpy
matplotlib
seaborn
Setup Instructions
Clone the Repository:
bash
Copy code
git clone https://github.com/your-username/parkinsons-disease-prediction.git
Install Requirements:
bash
Copy code
pip install -r requirements.txt
Run the Application:
bash
Copy code
python app.py
Open the Application: Visit http://127.0.0.1:5000/ in your web browser.
Usage
Navigate to the Web Interface: Open the application in your browser and enter the medical measurements as comma-separated values.

Get Predictions: Submit the data to receive predictions on Parkinson’s disease presence.

Model Information
Algorithm Used: Support Vector Machine (SVM) and Decision Tree
Metrics: Accuracy, Confusion Matrix, ROC Curve, Precision-Recall
Example Data
plaintext
Copy code
197.076,206.896,192.055,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.775,0.422229,0.741367,-7.3483,0.17755,1.743867,0.085569
Results
SVM Model: Outputs if a person has Parkinson’s Disease.
Decision Tree Model: Provides alternative prediction for comparison.
License
This project is licensed under the MIT License
