# src/models/grid_search.py
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import pickle
import os

# Define the base path
base_path = os.path.dirname(os.path.abspath(__file__))

# Load the datasets using the absolute path
X_train = pd.read_csv(os.path.join(base_path, '../../data/processed_data/X_train_scaled.csv'))
y_train = pd.read_csv(os.path.join(base_path, '../../data/processed_data/y_train.csv'))

# Define the model and parameters
model = LinearRegression()
param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}

# Perform GridSearch
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train.values.ravel())

# Save the best parameters using the absolute path
with open(os.path.join(base_path, '../../models/best_params.pkl'), 'wb') as file:
    pickle.dump(grid_search.best_params_, file)

