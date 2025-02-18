from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Define the path to the model file
MODEL_PATH = 'C:\\Users\\ARUNIMA\\OneDrive\\Desktop\\Team11_Virtual_Diagnosis of Skin Disorder_icv\\Ski_vi _project\\skin_disorder_model.h5'

# Load the pre-trained model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    print(f"Model file not found at {MODEL_PATH}")
    model = None  # Make sure to handle the case when model is not loaded

# Define allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route with upload form
@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' exists with an upload form

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model is not loaded", 500  # Return an error if the model isn't loaded

    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']

    if file and allowed_file(file.filename):
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)  # Save the uploaded file

        # Preprocess the image
        img = image.load_img(filepath, target_size=(150, 150))  # Adjust size to match model input
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # For simplicity, assume class labels; adjust for your trained classes
        class_labels = ['Acne', 'Eczema', 'Psoriasis', 'Rosacea']  # Customize this list
        result = class_labels[predicted_class[0]]

        # Dummy treatment information based on result (customize as needed)
        treatments = {
            'Acne': 'Use mild cleansers and avoid oily products.',
            'Eczema': 'Moisturize regularly and avoid allergens.',
            'Psoriasis': 'Use prescribed creams and light therapy.',
            'Rosacea': 'Avoid spicy foods and extreme temperatures.'
        }
        precautions = treatments.get(result, "Consult a dermatologist for advice.")

        return render_template('result.html', condition=result, precautions=precautions)

    return redirect(url_for('home'))

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)  # Create an 'uploads' folder if not existing
    app.run(debug=True)
