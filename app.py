from flask import Flask,render_template,request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import tf_keras as keras
import warnings

import os
from tensorflow_hub.keras_layer import KerasLayer


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Create a Flask instance
app = Flask(__name__)

model = keras.models.load_model('dog_vs_cat_classification.h5', custom_objects={'KerasLayer': KerasLayer})

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}),400
    
    img = request.files['image']

    # Save the uploaded image
    filename = 'image.jpg'
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img.save(img_path)
    input_image = cv2.imread(img_path)
    input_image_resize = cv2.resize(input_image,(224,224))
    input_image_scaled = input_image_resize/255
    image_reshaped = np.reshape(input_image_scaled,[1,224,224,3])
    input_prediction = model.predict(image_reshaped)
    input_pred_label = np.argmax(input_prediction)
    result = 'Cat' if input_pred_label == 0 else 'Dog'
    
    return render_template('result.html',filename=filename, result=result)
# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
