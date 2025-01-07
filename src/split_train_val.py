import os
import shutil
import random

# Paths to the train and validation directories
train_dir = '/app/dataset/PetImages/train'
validation_dir = '/app/dataset/PetImages/validation'

# Ensure train and validation directory exists
os.makedirs(validation_dir, exist_ok=True)

# Function to move a percentage of images from one directory to another
def move_images(train_class_dir, validation_class_dir, percentage):
    # List all images in the training class directory
    images = os.listdir(train_class_dir)
    # Calculate the number of images to move
    num_to_move = int(len(images) * percentage)
    # Randomly select images to move
    images_to_move = random.sample(images, num_to_move)
    
    # Ensure the validation class directory exists
    os.makedirs(validation_class_dir, exist_ok=True)
    
    for image in images_to_move:
        # Move the image
        shutil.move(os.path.join(train_class_dir, image), validation_class_dir)

# Define the class directories
classes = ['Cat', 'Dog']

# Move 10% of images from each class in train to validation
for class_name in classes:
    train_class_dir = os.path.join(train_dir, class_name)
    validation_class_dir = os.path.join(validation_dir, class_name)
    move_images(train_class_dir, validation_class_dir, percentage=0.10)

print("Moved 10% of images from train to validation.")
