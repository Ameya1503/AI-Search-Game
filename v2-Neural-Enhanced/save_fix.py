import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. Quick Load & Re-train (it will take only 1 minute with 10 epochs)
data = pd.read_csv('training_data.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
print("Running a quick final training to save the brain...")
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# 2. SAVE DIRECTLY TO C:\DeepCheckers (No 'models/' subfolder)
model.save('checkers_model.h5')
print("\nSuccess! Brain saved as 'checkers_model.h5' in your main folder.")