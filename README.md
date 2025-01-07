> This repository contains the code used in Lesson 4 of Module 2 of the Postgraduate course AI Technology Architect at PXL-Next.

## Requirements
Docker

## Usage

### Build container
`docker build -t pxl:lesson4 .`

### Train the classifier
`docker run -it --rm -v ./src:/app/src pxl:lesson4 python3.12 src/train_classifier.py`

### Use the classifier
This will print the path, the label and the confidence of a random image.

`docker run -it --rm -v ./src:/app/src pxl:lesson4 python3.12 src/classify.py`

### Show the image with the label and confidence
This will use the output of the classify script and show the image using pillow and matplotlib.

You will first need the dataset located at [keggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) unzipped and directory renamed to PetImages (with subfolders Cat and Dog)

`docker run -it --rm -v ./src:/app/src pxl:lesson4 python3.12 src/classify.py | xargs python3 src/show_image.py`
