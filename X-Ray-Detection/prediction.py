from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from pneumonia_prediction import predict_probability

app = Flask(__name__)

# Load the pre-trained TensorFlow model
# model = tf.keras.models.load_model('model.h5')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img = Image.open(img_file)
    image_name = '/images/img.png'
    image_file = f'./static{image_name}'
    img.save(image_file)
    print(img_file)
    print(image_file)
    pred_name = predict_probability(image_file)
    return render_template('predict.html', prediction=pred_name, image_file=image_name)

if __name__ == '__main__':
    app.run(debug=True)
