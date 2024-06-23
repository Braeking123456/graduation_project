from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import keras


# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = keras.models.load_model("BreastCancer_TrainModel.h5")

# Define a route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for uploading an image
@app.route('/upload', methods=['POST'])
def upload():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Load the image file
    image = Image.open(file)
    
    # You might want to perform additional processing here if needed
    
    # Return success response
    return jsonify({'message': 'Image uploaded successfully!'})

# Define a route for image classification
@app.route('/predict', methods=['POST'])
def classify():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Load the image file
    image = Image.open(file)
    
    # Resize and preprocess the image
    image = image.resize((150, 150))  # Resize the image to match the input size of your model
    image = np.array(image) / 255.0  # Normalize the pixel values

    # Perform prediction using the loaded model
    prediction = model.predict(np.expand_dims(image, axis=0))

    # Convert prediction probabilities to class labels
    class_names = ['benign', 'malignant', 'normal']  # Replace with your actual class names
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]

    return jsonify({'prediction': predicted_class_name})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
