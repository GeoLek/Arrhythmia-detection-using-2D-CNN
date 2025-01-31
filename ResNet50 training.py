import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# Base directory
output_base_dir = '/home/orion/Geo/Projects/2D CNN for arrthymia detection/data/'

# Create output directory
output_dir = os.path.join(output_base_dir, 'training_run_1')
os.makedirs(output_dir, exist_ok=True)

# File paths
model_checkpoint_path = os.path.join(output_dir, 'model_checkpoint.h5')
training_history_path = os.path.join(output_dir, 'training_history.csv')

# Training directory (no validation directory)
train_dir = '/home/orion/Geo/Projects/2D CNN for arrthymia detection/data/train'

# Generate file paths and labels for the specified classes
def generate_file_paths_and_labels(directory):
    file_paths = []
    labels = []
    for label in ['F', 'N', 'Q', 'S', 'V']:  # Specified class labels
        class_dir = os.path.join(directory, label)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            file_paths.append(file_path)
            labels.append(label)
    return file_paths, labels

train_image_paths, train_labels = generate_file_paths_and_labels(train_dir)

# Convert labels to categorical
labels_categorical = pd.get_dummies(train_labels).values

# Calculate class weights
labels_for_weights = np.argmax(labels_categorical, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(labels_for_weights), y=labels_for_weights)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Data generator for training, adjusted for grayscale images
train_datagen = ImageDataGenerator(rescale=1./255)

train_df = pd.DataFrame({'filename': train_image_paths, 'label': train_labels})

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,  # Using absolute paths
    x_col='filename',
    y_col='label',
    target_size=(224, 224),  # Match input size of ResNet
    color_mode='grayscale',  # Adjusted for grayscale images
    batch_size=32,
    class_mode='categorical',  # For multi-class classification
    shuffle=True,
)

# Modify ResNet50 model for grayscale input and multi-class output
input_tensor = Input(shape=(224, 224, 1))  # Grayscale images
base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)  # Corrected for 5 classes

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    epochs=10,
    class_weight=class_weight_dict,  # Apply class weights
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, monitor='accuracy', save_best_only=True, mode='max', verbose=1),
        tf.keras.callbacks.CSVLogger(training_history_path),
    ]
)

# Save the final model
model.save(os.path.join(output_dir, 'final_model.h5'))
