# src/models/model_training.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# Define the base path
base_path = os.path.dirname(os.path.abspath(__file__))

# Load the datasets using the absolute path
X_train = pd.read_csv(os.path.join(base_path, '../../data/processed_data/X_train_scaled.csv'))
y_train = pd.read_csv(os.path.join(base_path, '../../data/processed_data/y_train.csv'))

# Load the best parameters using the absolute path
with open(os.path.join(base_path, '../../models/best_params.pkl'), 'rb') as file:
    best_params = pickle.load(file)

# Train the model
model = LinearRegression(**best_params)
model.fit(X_train, y_train.values.ravel())

# Save the trained model using the absolute path
with open(os.path.join(base_path, '../../models/trained_model.pkl'), 'wb') as file:
    pickle.dump(model, file)

