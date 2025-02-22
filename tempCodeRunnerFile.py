
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score, RocCurveDisplay, PrecisionRecallDisplay
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as imbPipeline
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
data = pd.read_csv('C:\ML-aiproject\parkinson-disease-prediction-using-ml\parkinsons.csv')

# Preprocess the data
X = data.drop(columns=['name', 'status'])
y = data['status']

# Exploratory Data Analysis (EDA)
logging.info("Performing Exploratory Data Analysis (EDA)")
logging.info(f"Target variable distribution:\n{y.value_counts()}")
logging.info(f"Statistical summaries of features:\n{X.describe()}")

# Visualize feature distributions and correlations
plt.figure(figsize=(12, 8))
sns.pairplot(data, hue='status', diag_kind='kde')
plt.title('Pair Plot of Features')
plt.show()

plt.figure(figsize=(12, 8))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Address class imbalance using SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_res, y_res = smote_enn.fit_resample(X, y)

# Visualize the distribution of the target variable before and after SMOTEENN
plt.figure(figsize=(12, 6))
sns.countplot(x=y, label='Before SMOTEENN')
plt.title('Target Variable Distribution Before SMOTEENN')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x=y_res, label='After SMOTEENN')
plt.title('Target Variable Distribution After SMOTEENN')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Define the pipeline for SVM
pipeline_svm = imbPipeline(steps=[('scaler', StandardScaler()), ('classifier', SVC(probability=True))])

# Hyperparameter tuning for SVM
param_grid_svm = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [1, 0.1, 0.01, 0.001],
    'classifier__kernel': ['rbf', 'linear'],
    'classifier__class_weight': [None, 'balanced']
}

random_search_svm = RandomizedSearchCV(pipeline_svm, param_grid_svm, refit=True, verbose=2, cv=5, n_iter=50, random_state=42)
random_search_svm.fit(X_train, y_train)

# Best SVM model
svm_best = random_search_svm.best_estimator_

# Define the pipeline for Decision Tree
pipeline_dt = imbPipeline(steps=[('scaler', StandardScaler()), ('classifier', DecisionTreeClassifier())])

# Hyperparameter tuning for Decision Tree
param_grid_dt = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [None, 10, 20, 30, 40, 50],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__min_impurity_decrease': [0, 0.01, 0.05, 0.1],
    'classifier__class_weight': [None, 'balanced']
}

random_search_dt = RandomizedSearchCV(pipeline_dt, param_grid_dt, refit=True, verbose=2, cv=5, n_iter=50, random_state=42)
random_search_dt.fit(X_train, y_train)

# Best Decision Tree model
dt_best = random_search_dt.best_estimator_

# Define the pipeline for Random Forest
pipeline_rf = imbPipeline(steps=[('scaler', StandardScaler()), ('classifier', RandomForestClassifier())])

# Hyperparameter tuning for Random Forest
param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30, 40, 50],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__class_weight': [None, 'balanced']
}

random_search_rf = RandomizedSearchCV(pipeline_rf, param_grid_rf, refit=True, verbose=2, cv=5, n_iter=50, random_state=42)
random_search_rf.fit(X_train, y_train)

# Best Random Forest model
rf_best = random_search_rf.best_estimator_

# Define the pipeline for Gradient Boosting
pipeline_gb = imbPipeline(steps=[('scaler', StandardScaler()), ('classifier', GradientBoostingClassifier())])

# Hyperparameter tuning for Gradient Boosting
param_grid_gb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7, 10],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}

random_search_gb = RandomizedSearchCV(pipeline_gb, param_grid_gb, refit=True, verbose=2, cv=5, n_iter=50, random_state=42)
random_search_gb.fit(X_train, y_train)

# Best Gradient Boosting model
gb_best = random_search_gb.best_estimator_

# Evaluate the models
def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate the model's performance on the test set.

    Parameters:
    model: The trained model to evaluate.
    X_test: The test features.
    y_test: The test labels.
    threshold: The decision threshold for class predictions.

    Returns:
    accuracy: The accuracy score.
    f1: The F1 score.
    roc_auc: The ROC AUC score.
    conf_matrix: The confusion matrix.
    fpr: False positive rate.
    tpr: True positive rate.
    precision_vals: Precision values.
    recall_vals: Recall values.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)

    logging.info(f'Accuracy: {accuracy}')
    logging.info(f'Confusion Matrix:\n{conf_matrix}')
    logging.info(f'F1 Score: {f1}')
    logging.info(f'Precision: {precision}')
    logging.info(f'Recall: {recall}')
    logging.info(f'ROC AUC: {roc_auc}')

    return accuracy, f1, roc_auc, conf_matrix, fpr, tpr, precision_vals, recall_vals

# Evaluate SVM model
logging.info("Evaluating SVM Model:")
svm_accuracy, svm_f1, svm_roc_auc, svm_conf_matrix, svm_fpr, svm_tpr, svm_precision_vals, svm_recall_vals = evaluate_model(svm_best, X_test, y_test)

# Evaluate Decision Tree model
logging.info("Evaluating Decision Tree Model:")
dt_accuracy, dt_f1, dt_roc_auc, dt_conf_matrix, dt_fpr, dt_tpr, dt_precision_vals, dt_recall_vals = evaluate_model(dt_best, X_test, y_test)

# Evaluate Random Forest model
logging.info("Evaluating Random Forest Model:")
rf_accuracy, rf_f1, rf_roc_auc, rf_conf_matrix, rf_fpr, rf_tpr, rf_precision_vals, rf_recall_vals = evaluate_model(rf_best, X_test, y_test)

# Evaluate Gradient Boosting model
logging.info("Evaluating Gradient Boosting Model:")
gb_accuracy, gb_f1, gb_roc_auc, gb_conf_matrix, gb_fpr, gb_tpr, gb_precision_vals, gb_recall_vals = evaluate_model(gb_best, X_test, y_test)

# Save the best models
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_best, file)

with open('dt_model.pkl', 'wb') as file:
    pickle.dump(dt_best, file)

with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf_best, file)

with open('gb_model.pkl', 'wb') as file:
    pickle.dump(gb_best, file)

# Extract and save the scaler from the best model pipeline
scaler = svm_best.named_steps['scaler']
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Feature importance from Decision Tree
feature_importances = dt_best.named_steps['classifier'].feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
logging.info("Feature Importances from Decision Tree:")
logging.info(feature_importance_df)

# Plot Feature Importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances from Decision Tree')
plt.show()

# Visualize correlations between features
plt.figure(figsize=(12, 8))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Analyze outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=X)
plt.xticks(rotation=90)
plt.title('Boxplot of Features')
plt.show()

# Model comparison summary
models = ['SVM', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
accuracies = [svm_accuracy, dt_accuracy, rf_accuracy, gb_accuracy]
f1_scores = [svm_f1, dt_f1, rf_f1, gb_f1]
roc_aucs = [svm_roc_auc, dt_roc_auc, rf_roc_auc, gb_roc_auc]

summary_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'F1 Score': f1_scores,
    'ROC AUC': roc_aucs
})

logging.info("Model Comparison Summary:")
logging.info(summary_df)

# Plot Model Comparison
plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='Accuracy', data=summary_df)
plt.title('Model Accuracy Comparison')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='F1 Score', data=summary_df)
plt.title('Model F1 Score Comparison')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='ROC AUC', data=summary_df)
plt.title('Model ROC AUC Comparison')
plt.show()

# Threshold Tuning
thresholds = np.arange(0.1, 1.0, 0.1)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    _, f1, _, _, _, _, _, _ = evaluate_model(gb_best, X_test, y_test, threshold)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

logging.info(f"Best Threshold: {best_threshold}")
logging.info(f"Best F1 Score: {best_f1}")

# Model Ensembling using Stacking
estimators = [
    ('svm', svm_best.named_steps['classifier']),
    ('dt', dt_best.named_steps['classifier']),
    ('rf', rf_best.named_steps['classifier']),
    ('gb', gb_best.named_steps['classifier'])
]

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_clf.fit(X_train, y_train)

# Evaluate Stacking model
logging.info("Evaluating Stacking Model:")
stacking_accuracy, stacking_f1, stacking_roc_auc, stacking_conf_matrix, stacking_fpr, stacking_tpr, stacking_precision_vals, stacking_recall_vals = evaluate_model(stacking_clf, X_test, y_test)

# Add Stacking model to comparison
models.append('Stacking')
accuracies.append(stacking_accuracy)
f1_scores.append(stacking_f1)
roc_aucs.append(stacking_roc_auc)

summary_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'F1 Score': f1_scores,
    'ROC AUC': roc_aucs
})

logging.info("Model Comparison Summary with Stacking:")
logging.info(summary_df)

# Plot Model Comparison with Stacking
plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='Accuracy', data=summary_df)
plt.title('Model Accuracy Comparison with Stacking')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='F1 Score', data=summary_df)
plt.title('Model F1 Score Comparison with Stacking')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='ROC AUC', data=summary_df)
plt.title('Model ROC AUC Comparison with Stacking')
plt.show()

# Plot ROC Curves for all models
plt.figure(figsize=(12, 8))
RocCurveDisplay.from_estimator(svm_best, X_test, y_test, name='SVM')
RocCurveDisplay.from_estimator(dt_best, X_test, y_test, name='Decision Tree')
RocCurveDisplay.from_estimator(rf_best, X_test, y_test, name='Random Forest')
RocCurveDisplay.from_estimator(gb_best, X_test, y_test, name='Gradient Boosting')
RocCurveDisplay.from_estimator(stacking_clf, X_test, y_test, name='Stacking')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curves')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall Curves for all models
plt.figure(figsize=(12, 8))
PrecisionRecallDisplay.from_estimator(svm_best, X_test, y_test, name='SVM')
PrecisionRecallDisplay.from_estimator(dt_best, X_test, y_test, name='Decision Tree')
PrecisionRecallDisplay.from_estimator(rf_best, X_test, y_test, name='Random Forest')
PrecisionRecallDisplay.from_estimator(gb_best, X_test, y_test, name='Gradient Boosting')
PrecisionRecallDisplay.from_estimator(stacking_clf, X_test, y_test, name='Stacking')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc="lower left")
plt.show()

# Plot Learning Curves for all models
plt.figure(figsize=(12, 8))
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curves')

# SVM
train_sizes, train_scores, test_scores = learning_curve(svm_best, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='SVM Training Score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='SVM Cross-Validation Score')

# Decision Tree
train_sizes, train_scores, test_scores = learning_curve(dt_best, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='Decision Tree Training Score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='c', label='Decision Tree Cross-Validation Score')

# Random Forest
train_sizes, train_scores, test_scores = learning_curve(rf_best, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, 'o-', color='m', label='Random Forest Training Score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='y', label='Random Forest Cross-Validation Score')

# Gradient Boosting
train_sizes, train_scores, test_scores = learning_curve(gb_best, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, 'o-', color='k', label='Gradient Boosting Training Score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Gradient Boosting Cross-Validation Score')

plt.legend(loc="best")
plt.show()
