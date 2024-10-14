Parkinson's Disease Detection README
Overview
This project implements a machine learning model to detect Parkinson's Disease based on vocal features. The dataset used for this project is obtained from a CSV file containing various attributes related to voice recordings of individuals diagnosed with Parkinson's Disease.
Table of Contents
Technologies Used
Dataset
Installation
Usage
Model Training
Model Evaluation
Predictive System
Contributing
License
Technologies Used
Python
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Dataset
The dataset used in this project is a CSV file (parkinsons.csv) that contains the following columns:
MDVP:Fo(Hz): Fundamental frequency
MDVP:Fhi(Hz): Maximum frequency
MDVP:Flo(Hz): Minimum frequency
MDVP:Jitter(%): Jitter percentage
MDVP:Jitter(Abs): Absolute jitter
MDVP:RAP: Relative average perturbation
MDVP:PPQ: Pitch period perturbation quotient
Shimmer: Amplitude variation
NHR: Noise-to-harmonics ratio
HNR: Harmonics-to-noise ratio
RPDE: Recurrence period density entropy
DFA: Detrended fluctuation analysis
status: Target variable (1 = Parkinson's positive, 0 = Healthy)
Installation
To run this project, ensure you have Python installed on your machine. Then, install the required libraries using pip:
bash
pip install numpy pandas scikit-learn matplotlib seaborn

Usage
Clone the repository or download the files.
Place the parkinsons.csv file in the same directory as the script.
Run the script using Python:
bash
python copy_of_project2_parkinson-s_disease_detection-rm.py

Model Training
The project includes two machine learning models for classification:
Support Vector Machine (SVM)
Decision Tree Classifier
Both models are trained using a training dataset split from the original dataset.
Steps for Model Training:
Load and preprocess the data.
Split the data into training and testing sets.
Standardize the features.
Train both models on the training data.
Model Evaluation
After training, the models are evaluated based on their accuracy scores and confusion matrices:
Calculate accuracy on both training and testing datasets.
Generate confusion matrices to visualize model performance.
Predictive System
The script includes functionality to make predictions based on new input data. Users can input vocal feature values, and the model will predict whether the individual has Parkinson's Disease or not.
