# Dog vs Cat Classification
This is a simple web application built with Flask, TensorFlow, and OpenCV to classify images as either a dog or a cat. The model used for classification is a Convolutional Neural Network (CNN) trained on a dog vs cat dataset.

## Vist here
 - https://dog-vs-cat-classification.onrender.com/

## Features
- Upload an image (PNG, JPG, or JPEG format) of a dog or a cat.
- The model classifies the image as either "Dog" or "Cat".
- Flask-based web interface for user interaction.
### Tech Stack
- Backend: Flask (Python web framework)
- Machine Learning: TensorFlow, Keras
- Image Processing: OpenCV
- Deployment: Heroku (or any other preferred platform)
## Prerequisites
Before running the application, make sure you have the following installed:

- Python 3.7+
- pip (Python package installer)
## Installation
Follow these steps to run the project on your local machine:

1. Clone the repository:
  - [git clone https://github.com/your-username/dog-vs-cat-classification.git](https://github.com/sumanthgubbala/Dog-vs-cat-classification.git)

2. Navigate to the project directory:
 - cd dog-vs-cat-classification
3. Create and activate a virtual environment (optional but recommended):
 - On Windows:
    - python -m venv venv
    - .\venv\Scripts\activate
- On macOS/Linux:
    - python -m venv venv
    - source venv/bin/activate
4. Install the required dependencies:
    - pip install -r requirements.txt
## Running the Application
To run the app locally, use the following command:
   - python app.py
   - Visit http://127.0.0.1:5000 in your browser to interact with the application.

## How to Use
1. Open the application in your web browser.
2. Upload an image of a dog or a cat.
3. The model will process the image and display whether the image is a "Dog" or "Cat".
## Model Training
The model used for classification is a pre-trained Keras model (dog_vs_cat_classification.h5). If you want to retrain the model, follow these steps:

1. Gather a dataset of dog and cat images.
2. Preprocess the images (resize to 224x224 and normalize).
3. Train a Convolutional Neural Network (CNN) on the dataset using Keras/TensorFlow.
4. Save the model as dog_vs_cat_classification.h5 and place it in the project directory.
