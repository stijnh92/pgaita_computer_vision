import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random
import pathlib

# Load the trained model
model = tf.keras.models.load_model('/app/src/cats_and_dogs_classifier_new.h5', compile=False)

# Path to the validation directory
validation_dir = '/app/dataset/PetImages/validation'
classes = ['Cat', 'Dog']

# Collect all images from the validation set
image_paths = []
for class_name in classes:
    class_dir = os.path.join(validation_dir, class_name)
    image_paths += [os.path.join(class_dir, img) for img in os.listdir(class_dir)]

# Shuffle images to mix cats and dogs
random.shuffle(image_paths)

# Function to load and preprocess an image
def preprocess_image(image_path):
    # Open the image file
    img = Image.open(image_path)
    
    # Resize the image to 150x150
    img = img.resize((150, 150))
    
    # Convert the image to a numpy array and rescale to [0, 1]
    img = np.array(img) / 255.0
    
    # Ensure the array has three channels (RGB)
    if len(img.shape) == 2:  # Grayscale image
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:  # RGBA image
        img = img[:, :, :3]
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# Iterate over images and show predictions
img = preprocess_image(image_paths[0])
prediction = model.predict(img, verbose=0)[0][0]

path = pathlib.PurePath(image_paths[0])
label = 'Dog' if prediction > 0.5 else 'Cat'
confidence = prediction if label == 'Dog' else 1 - prediction

print(f"{path.parent.name}/{path.name} {label} {confidence:.2f}")
