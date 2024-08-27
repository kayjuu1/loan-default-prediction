import joblib
import numpy as np

# Load the trained model
model = joblib.load('knn_model.pkl')

# Example input with 7 features (modify this to include appropriate values for the missing features)
input_data = np.array([[5.1, 3.5, 1.4, 0.2, 0.0, 0.0, 0.0]])  # Ensure it has 7 features

# Make prediction
prediction = model.predict(input_data)

print(f"Prediction: {prediction}")
