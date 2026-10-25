
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Dataset and model paths
DATASET_PATH = (
    r"C:\Users\HP\Downloads\PlantVillage-Dataset-master\raw\color"
)
MODEL_PATH = "plant_disease_output/plant_disease_cnn.keras"

# Load trained model
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")
print("Model output classes:", model.output_shape[-1])

# Get class names in the same order used during training
class_names = sorted(
    folder.name
    for folder in Path(DATASET_PATH).iterdir()
    if folder.is_dir()
)

print("Number of classes:", len(class_names))

# Enter image path
image_path = input("Enter the leaf image path: ").strip().strip('"')

if not Path(image_path).is_file():
    print("Error: Image file not found. Check the path.")
    raise SystemExit

# Load and preprocess image
img = keras.utils.load_img(
    image_path,
    target_size=(128, 128)
)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array, verbose=0)[0]
predicted_index = int(tf.argmax(predictions))

print("\nPredicted disease:", class_names[predicted_index])
print(
    "Model confidence:",
    f"{float(predictions[predicted_index]) * 100:.2f}%"
)
