FROM python:3.12.8

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    unzip \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install numpy opencv-python matplotlib tensorflow scipy

# Download and unzip the cats and dogs dataset using curl
RUN curl -L -o /home/microsoft-catsvsdogs-dataset.zip \
       "https://www.kaggle.com/api/v1/datasets/download/shaunthesheep/microsoft-catsvsdogs-dataset" \
    && unzip /home/microsoft-catsvsdogs-dataset.zip -d /app/dataset \
    && rm /home/microsoft-catsvsdogs-dataset.zip

# Create the train directory and move Dog and Cat directory
RUN mkdir -p /app/dataset/PetImages/train \
    && mv /app/dataset/PetImages/Dog /app/dataset/PetImages/train/Dog \
    && mv /app/dataset/PetImages/Cat /app/dataset/PetImages/train/Cat

# Noticed some issues with two files in the dataset while training the classifier
RUN rm /app/dataset/PetImages/train/Dog/11702.jpg
RUN rm /app/dataset/PetImages/train/Cat/666.jpg

# Split the dataset in training and validation data
COPY src/split_train_val.py /app/split_train_val.py
RUN python3.12 split_train_val.py
RUN rm /app/split_train_val.py
