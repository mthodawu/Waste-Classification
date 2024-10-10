from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('model/your_model.keras')

# Define allowed file extensions for images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Prepare image and make prediction
            img_array = prepare_image(file_path, target_size=(224, 224))  # Adjust size as per your model
            prediction = model.predict(img_array)

            # Process prediction and return result
            result = np.argmax(prediction, axis=1)[0]  # Or any other post-processing you need
            return render_template('result.html', prediction=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
