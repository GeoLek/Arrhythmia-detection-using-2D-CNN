import os
import random
import shutil

# Define your dataset directory and the output directories for splits
dataset_dir = '/home/orion/Geo/Projects/2D CNN for arrthymia detection/ECG_Image_data/data/V'
output_dir = '/home/orion/Geo/Projects/2D CNN for arrthymia detection/ECG_Image_data/Split'

# Directories for train and test splits
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

# Make sure directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read all file names directly from the dataset directory
file_list = [f for f in os.listdir(dataset_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Apply a more random shuffle
random.shuffle(file_list)

# Define split ratios
train_ratio = 0.8  # Adjusted to 80% for training
# The remaining 20% will be for test_ratio

# Calculate split indices
train_split = int(train_ratio * len(file_list))

# Move files to the respective directories
for i, file_name in enumerate(file_list):
    src_path = os.path.join(dataset_dir, file_name)

    if i < train_split:
        dst_path = os.path.join(train_dir, file_name)
    else:
        dst_path = os.path.join(test_dir, file_name)

    shutil.move(src_path, dst_path)

print(f"Dataset split: {train_ratio * 100}% train, {100 - (train_ratio * 100)}% test.")
