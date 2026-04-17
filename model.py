import numpy as np
import tensorflow as tf
from tensorflow import keras

# Create simple ANN model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=[1]),
    keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Training function
def train(X, y, epochs=500):
    history = model.fit(X, y, epochs=epochs, verbose=0)
    return history

# Prediction function
def predict(celsius):
    return model.predict([celsius], verbose=0)[0][0]

# Save/Load functions
def save_model(path='model.h5'):
    model.save(path)

def load_model(path='model.h5'):
    global model
    model = keras.models.load_model(path)

# Train and save if run directly
if __name__ == '__main__':
    # Generate training data
    celsius = np.random.uniform(-100, 100, 1000)
    fahrenheit = celsius * 1.8 + 32
    
    # Train model
    print("Training model...")
    train(celsius, fahrenheit, epochs=500)
    
    # Save model
    save_model()
    print("Model saved as model.h5")
    
    # Test
    test_celsius = [0, 25, 37, 100, -40]
    for c in test_celsius:
        pred = predict(c)
        actual = c * 1.8 + 32
        print(f"{c}°C = {pred:.2f}°F (Actual: {actual:.2f}°F)")
