import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import math
import time

base_path = '/home/orion/Geo/Projects/2D CNN for arrthymia detection/ECG_Image_data/data'
output_path = '/home/orion/Geo/Projects/2D CNN for arrthymia detection/ECG_Image_data/augmented'

augmentation = ImageDataGenerator(
    rotation_range=0.1,
    width_shift_range=0.01,
    height_shift_range=0.01,
    shear_range=0.01,
    zoom_range=0.01,
    horizontal_flip=False,
    fill_mode='nearest'
)

target_dir = os.path.join(base_path, 'F')
output_dir = os.path.join(output_path, 'F')
os.makedirs(output_dir, exist_ok=True)

current_image_count = len([name for name in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, name))])
goal_image_count = 100000
required_augmentations = goal_image_count - current_image_count

num_augmented_images_per_original = math.ceil(required_augmentations / current_image_count)

augmented_count = 0

for filename in os.listdir(target_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        file_base = os.path.splitext(filename)[0]  # Get filename without extension
        image_path = os.path.join(target_dir, filename)
        image = load_img(image_path)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        i = 0
        for batch in augmentation.flow(image, batch_size=1, save_prefix=file_base + "_" + str(int(time.time())), save_to_dir=output_dir, save_format='png'):
            i += 1
            augmented_count += 1
            if i >= num_augmented_images_per_original or augmented_count >= required_augmentations:
                break  # Stop if we reach the required number of augmentations

        if augmented_count >= required_augmentations:
            break  # Exit the outer loop if we've generated enough augmentations

# Note: This adjusts for any potential overshoot in required augmentations
