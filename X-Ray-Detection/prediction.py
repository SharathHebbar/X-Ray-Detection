from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from pneumonia_prediction import predict_probability

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    image_name = '/images/img.png'
    image_file = f'./static{image_name}'
    img_file.save(image_file)
    pred_name = predict_probability(image_file)
    return render_template('predict.html', prediction=pred_name, image_file=image_name)

if __name__ == '__main__':
    app.run(debug=True)
