import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os

# 1. Load the dataset you generated
file_path = 'training_data.csv'
if not os.path.exists(file_path):
    print("Error: training_data.csv not found! Run collector.py first.")
    exit()

data = pd.read_csv(file_path, header=None)
X = data.iloc[:, :-1].values  # First 64 columns (Board squares)
y = data.iloc[:, -1].values   # Last column (Heuristic Score)

# 2. Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define the Deep MLP Architecture
# We use 3 hidden layers to ensure it qualifies as a 'Deep' Neural Network
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(64,)), # Input layer
    layers.Dense(64, activation='relu'),                     # Hidden Layer 1
    layers.Dense(32, activation='relu'),                     # Hidden Layer 2
    layers.Dense(1)                                          # Output (Predicted Score)
])

# 4. Compile the model
# Using 'mse' (Mean Squared Error) because we are predicting a numerical score
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 5. Train the Model
print("Training the Deep Neural Network...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 6. Save the trained 'Brain'
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/checkers_model.h5')
print("\nSuccess! Model saved to models/checkers_model.h5")