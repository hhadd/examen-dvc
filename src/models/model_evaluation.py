# src/models/model_evaluation.py
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

# Define the base path
base_path = os.path.dirname(os.path.abspath(__file__))

# Load the datasets using the absolute path
X_test = pd.read_csv(os.path.join(base_path, '../../data/processed_data/X_test_scaled.csv'))
y_test = pd.read_csv(os.path.join(base_path, '../../data/processed_data/y_test.csv'))

# Load the trained model using the absolute path
with open(os.path.join(base_path, '../../models/trained_model.pkl'), 'rb') as file:
    model = pickle.load(file)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Save the predictions using the absolute path
pd.DataFrame(predictions, columns=['predictions']).to_csv(os.path.join(base_path, '../../data/processed_data/predictions.csv'), index=False)

# Save the evaluation metrics using the absolute path
metrics = {'MSE': mse, 'R2': r2}
with open(os.path.join(base_path, '../../metrics/scores.json'), 'w') as file:
    json.dump(metrics, file)

