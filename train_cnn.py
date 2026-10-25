
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Dataset path
DATASET_PATH = (
    r"C:\Users\HP\Downloads\PlantVillage-Dataset-master\raw\color"
)

# Training configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
SEED = 42

# Load images in batches; don't load all images into RAM
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Number of classes:", num_classes)
print("Classes:", class_names)

# Normalize pixels from 0-255 to 0-1
normalization = layers.Rescaling(1.0 / 255)

# Build a small Convolutional Neural Network
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    normalization,

    layers.Conv2D(16, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save the best model based on validation loss
output_dir = Path("plant_disease_output")
output_dir.mkdir(exist_ok=True)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(output_dir / "best_model.keras"),
    monitor="val_loss",
    save_best_only=True
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train for up to 10 epochs
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[checkpoint, early_stop]
)

# Save the final model
model.save(output_dir / "plant_disease_cnn.keras")

print("Training complete!")
print("Model saved in:", output_dir.resolve())
