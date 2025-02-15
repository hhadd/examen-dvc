# src/data/data_normalization.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Define the base path
base_path = os.path.dirname(os.path.abspath(__file__))

# Load the datasets using the absolute path
X_train = pd.read_csv(os.path.join(base_path, '../../data/processed_data/X_train.csv'))
X_test = pd.read_csv(os.path.join(base_path, '../../data/processed_data/X_test.csv'))

# Normalize the datasets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaled datasets using the absolute path
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(os.path.join(base_path, '../../data/processed_data/X_train_scaled.csv'), index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(os.path.join(base_path, '../../data/processed_data/X_test_scaled.csv'), index=False)

