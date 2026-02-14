import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("Generating visualization data...")
data = pd.read_csv('training_data.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
model = tf.keras.models.load_model('checkers_model.h5', compile=False)


y_pred = model.predict(X[:500]) 
plt.figure(figsize=(8, 6))
plt.scatter(y[:500], y_pred, alpha=0.5, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--') 
plt.title('Neural Optimization: MLP Prediction vs. Classical Heuristic')
plt.xlabel('Original Heuristic Score (Classical)')
plt.ylabel('MLP Predicted Score (Neural)')
plt.grid(True)
plt.savefig('correlation_plot.png')
print("Saved: correlation_plot.png")

weights = np.abs(model.layers[0].get_weights()[0])
square_importance = np.mean(weights, axis=1).reshape(8, 8)

plt.figure(figsize=(8, 6))
plt.imshow(square_importance, cmap='hot', interpolation='nearest')
plt.colorbar(label='Weight Intensity')
plt.title('Feature Importance: Which squares does the AI value?')
plt.xticks(range(8), range(1, 9))
plt.yticks(range(8), range(1, 9))
plt.savefig('board_heatmap.png')
print("Saved: board_heatmap.png")

print("\nVisualization Complete! Check your folder for the .png files.")