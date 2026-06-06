import joblib
import numpy as np

# Load saved model and encoder
model = joblib.load("models/crop_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Example input:
# N, P, K, temperature, humidity, ph, rainfall
sample_data = np.array([[90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]])

# Predict
prediction = model.predict(sample_data)
predicted_crop = label_encoder.inverse_transform(prediction)

print("Recommended Crop:", predicted_crop[0])