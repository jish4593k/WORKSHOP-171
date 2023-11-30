import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Generate some sample data for regression
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X.squeeze() + 1 + 0.1 * np.random.randn(100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model using scikit-learn
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Create a simple neural network using TensorFlow and Keras
model = keras.Sequential([
    keras.layers.Dense(units=1, input_dim=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# GUI Setup
def predict_linear_regression():
    x_value = float(entry.get())
    linear_prediction = linear_model.predict([[x_value]])
    linear_result.set(f"Linear Regression Prediction: {linear_prediction[0]:.2f}")

def predict_neural_network():
    x_value = float(entry.get())
    nn_prediction = model.predict([x_value])
    nn_result.set(f"Neural Network Prediction: {nn_prediction[0][0]:.2f}")

# Create the main window
root = tk.Tk()
root.title("Regression Prediction GUI")

# Add a label and entry for input value
label = ttk.Label(root, text="Enter X value:")
label.pack(pady=10)
entry = ttk.Entry(root)
entry.pack(pady=10)

# Buttons for predictions
linear_result = tk.StringVar()
nn_result = tk.StringVar()

linear_button = ttk.Button(root, text="Linear Regression Predict", command=predict_linear_regression)
linear_button.pack(pady=10)
linear_result_label = ttk.Label(root, textvariable=linear_result)
linear_result_label.pack(pady=10)

nn_button = ttk.Button(root, text="Neural Network Predict", command=predict_neural_network)
nn_button.pack(pady=10)
nn_result_label = ttk.Label(root, textvariable=nn_result)
nn_result_label.pack(pady=10)

# Run the GUI
root.mainloop()
