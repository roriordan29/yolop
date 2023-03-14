import os
import random
import shutil

# Set the paths to the source and destination directories
src_dir = 'Data'
train_dir = 'train'
val_dir = 'test'
test_dir = 'val'

# Set the percentage split between the sets
train_percent = 0.6
val_percent = 0.2
test_percent = 0.2

# Get a list of all image file names in the source directory
image_files = os.listdir(src_dir)

# Shuffle the list of image files randomly
random.shuffle(image_files)

# Calculate the number of images for each set based on the percentage split
num_train = int(len(image_files) * train_percent)
num_val = int(len(image_files) * val_percent)
num_test = int(len(image_files) * test_percent)

# Divide the shuffled image files into three sets
train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]
test_files = image_files[num_train + num_val:]

# Move the image files into their respective directories
for file in train_files:
    shutil.move(os.path.join(src_dir, file), os.path.join(train_dir, file))

for file in val_files:
    shutil.move(os.path.join(src_dir, file), os.path.join(val_dir, file))

for file in test_files:
    shutil.move(os.path.join(src_dir, file), os.path.join(test_dir, file))

