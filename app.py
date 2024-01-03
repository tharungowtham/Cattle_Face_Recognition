"""
app.py - Flask web application for cow identification and registration.

This application uses a Siamese neural network model to identify and register cows.
It allows users to add cow details, including owner information and an image of the cow.
Users can also identify a cow by uploading an image, and the application checks for
similarity with previously registered cows.

"""

from __future__ import annotations
from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
from pymongo import MongoClient
import base64
import os
import tensorflow as tf
import tensorflow as tf
import keras
import PIL
from io import BytesIO
import random

# Initialize Flask app
app = Flask(__name__)

# Connect to MongoDB database
client = MongoClient("mongodb+srv://varun:varun@mooo.vgkvsiy.mongodb.net/")
db = client.mooo
cow_history = db.cow_history

# Load Siamese neural network model and extract necessary layers
loaded_model = tf.keras.models.load_model('./siamesemodelsoftmax')
encoder_layer = keras.Model(inputs=loaded_model.input, outputs=loaded_model.get_layer("concatenate").output)
classifier = keras.Model(inputs=encoder_layer.output, outputs=loaded_model.output)

def similarity(encoded_repr_1: np.ndarray, encoded_repr_2: np.ndarray) -> tuple[bool, float]:
    """
    Calculate similarity between two encoded representations using a classifier.

    Args:
        encoded_repr_1 (np.ndarray): Encoded representation of the first input.
        encoded_repr_2 (np.ndarray): Encoded representation of the second input.

    Returns:
        tuple[bool, float]: A tuple containing a boolean indicating similarity and a similarity score.
    """
    probabilities = classifier.predict(np.array([np.concatenate([encoded_repr_1, encoded_repr_2])]))
    similar, different = probabilities[0]
    return similar >= 0.95, similar

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/addcow', methods=['POST', 'GET'])
def add_cow_to_db():
    """
    Add cow details to the MongoDB database.

    If the uploaded cow image is similar to any existing cow, it notifies the user.

    Returns:
        Flask template: Rendered template based on the registration result.
    """
    if request.method == 'POST':
        # Extract owner data from the form
        owner_data = {
            'owner_name': request.form['name'],
            'owner_phno': request.form['phno'],
            'owner_address': request.form['add'],
            'city': request.form['city'],
            'state': request.form['state'],
            'zip': request.form['zip'],
        }

        # Process the uploaded cow image
        image = request.files['image']
        cow_image = PIL.Image.open(image)
        cow_image = cow_image.convert("RGB")
        cow_image = cow_image.resize((224, 224))
        image_data = np.asarray(cow_image).reshape(1, 224, 224, 3).astype('float32') / 255

        # Extract encoded representation using the Siamese model
        output = encoder_layer.predict([image_data, image_data])
        encoded_representation = output[0][:3136]

        # Check for similarity with existing cows
        for data in cow_history.find({}):
            is_similar, similarity_score = similarity(np.array(data['encoded_rep'], dtype="float32"), encoded_representation)
            if is_similar:
                return render_template("register.html", exists=True, cow_data=data, similarity_score=similarity_score * 100)

        # Save cow details to the database
        image_bytes_io = BytesIO()
        cow_image.save(image_bytes_io, format='JPEG')
        image_bytes = image_bytes_io.getvalue()
        B64EI = base64.b64encode(image_bytes).decode('utf-8')
        cow_history.insert_one({
            **owner_data,
            'cow_image': B64EI,
            'encoded_rep': list(encoded_representation.astype('object')),
        })
        cow_data = cow_history.find_one(sort=[('_id', -1)])
        return render_template('register.html', exists=False, cow_data=cow_data)
    else:
        return redirect("/")

@app.route('/identify', methods=['POST'])
def identify_the_cow():
    """
    Identify a cow based on the uploaded image.

    If the cow is identified, it notifies the user; otherwise, it allows registration.

    Returns:
        Flask template: Rendered template based on the identification result.
    """
    if 'image' in request.files:
        file = request.files['image']
        if file:
            # Process the uploaded cow image
            cow_image = PIL.Image.open(file)
            cow_image = cow_image.convert("RGB")
            image_bytes_io = BytesIO(cow_image.tobytes())
            B64EI = base64.b64encode(image_bytes_io.getvalue()).decode('utf-8')
            cow_image = cow_image.resize((224, 224))
            image_data = np.asarray(cow_image).reshape(1, 224, 224, 3).astype('float32') / 255

            # Extract encoded representation using the Siamese model
            output = encoder_layer.predict([image_data, image_data])
            encoded_representation = output[0][:3136]

            # Check for similarity with existing cows
            for data in cow_history.find({}):
                is_similar, similarity_score = similarity(encoded_representation, np.array(data['encoded_rep'], dtype='float32'))
                if is_similar:
                    return render_template('identify.html', exists=True, similarity_score=similarity_score * 100, cow_data=data)

            # If no similar cow is found, allow registration
            return render_template('identify.html', exists=False, base64_encoding=B64EI)

    return redirect('/')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory(app['staticdir'], path)

if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)
