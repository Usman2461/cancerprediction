from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Add this first
import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('skin_cancer_model.h5')

# Define class labels (adjust based on your dataset)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((128, 128))  # Match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))  # Fixed line
    image = preprocess_image(image)  # Preprocess

    # Make prediction
    prediction = model.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        'prediction': predicted_class,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)