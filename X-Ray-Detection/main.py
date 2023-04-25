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
    file = request.files['image']
    # show the loading screen
    return """
    <html>
    <head>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
        <script>
            $(document).ready(function() {
                // make an AJAX request to the server to start the prediction
                $.ajax({
                    type: 'POST',
                    url: '/do_prediction',
                    data: new FormData($('form')[0]),
                    processData: false,
                    contentType: false,
                    success: function(data) {
                        // hide the loading screen and display the result
                        hideLoading();
                        $('#result').html(data);
                    },
                    error: function(xhr, status, error) {
                        alert('Error: ' + error.message);
                    }
                });
            });
        </script>
    </head>
    <body>
        <div id="loading">Loading...</div>
        <div id="result"></div>
    </body>
    </html>
    """

@app.route('/do_prediction', methods=['POST'])
def do_prediction():
    # file = request.files['image']
    # perform the prediction here
    result = "Prediction complete!"
    return result


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
