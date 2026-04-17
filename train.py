from model import train, save_model, predict
import numpy as np

# Generate data
print("Generating temperature data...")
celsius = np.random.uniform(-100, 100, 2000)
fahrenheit = celsius * 1.8 + 32

# Train model
print("Training neural network...")
train(celsius, fahrenheit, epochs=500)

# Save model
save_model('celsius_to_fahrenheit.h5')
print("Model saved as celsius_to_fahrenheit.h5")

# Test the model
print("\nTesting model:")
test_values = [0, 25, 37, 100, -40]
for c in test_values:
    pred = predict(c)
    actual = c * 1.8 + 32
    error = abs(pred - actual)
    print(f"{c:3}°C -> {pred:6.2f}°F (Actual: {actual:6.2f}°F, Error: {error:.4f}°F)")
