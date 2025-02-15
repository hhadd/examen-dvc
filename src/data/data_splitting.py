# src/data/data_splitting.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define the base path
base_path = os.path.dirname(os.path.abspath(__file__))

# Load the dataset using the absolute path
data = pd.read_csv(os.path.join(base_path, '../../data/raw_data/raw.csv'))

# Drop the first date column
data = data.drop(columns=[data.columns[0]])

# Split the dataset
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the datasets using the absolute path
processed_path = os.path.join(base_path, '../../data/processed_data/')
X_train.to_csv(os.path.join(processed_path, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(processed_path, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(processed_path, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(processed_path, 'y_test.csv'), index=False)

