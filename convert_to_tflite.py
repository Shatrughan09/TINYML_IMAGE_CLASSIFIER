import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("saved_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimization (quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created successfully!")
