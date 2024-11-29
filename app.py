from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Configure file upload directory
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the crop/weed detection model
try:
    model_path = 'C:/Users/mihir/Desktop/Crop and weed detection/models/crop_weed_model.keras'
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to preprocess images
def preprocess_image(image):
    try:
        print("Original image size:", image.size)
        image = image.resize((224, 224))  # Resize to model's expected input size
        image_array = np.array(image) / 255.0  # Normalize the image
        print("Image array shape:", image_array.shape)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# Route for the home page (crop/weed detection)
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    img_path = None
    error_message = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            error_message = "Invalid file uploaded. Please upload a PNG, JPG, or JPEG image."
        else:
            try:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(file_path)
                print(f"File saved to: {file_path}")

                img = Image.open(file_path)
                img_array = preprocess_image(img)

                if img_array is None:
                    error_message = "Error processing the uploaded image."
                else:
                    # Get prediction from the model (binary classification: crop or weed)
                    prediction = model.predict(img_array)
                    print("Model prediction:", prediction)

                    # Interpret the prediction
                    result = 'crop' if prediction[0][0] < 0.5 else 'weed'
                    img_path = f"/{file_path.replace('\\', '/')}"

            except Exception as e:
                error_message = f"An error occurred during prediction: {e}"
                print(error_message)

    return render_template('index.html', result=result, img_path=img_path, error_message=error_message)

# API route for chatbot functionality
@app.route('/get_answer', methods=['POST'])
def get_answer():
    context = """
    Crops are plants cultivated for food, fodder, fiber, and fuel. Weeds are unwanted plants that grow among crops 
    and compete for nutrients, water, and sunlight. Effective weed control involves techniques like manual removal, 
    chemical herbicides, and modern AI detection methods. Common crops include wheat, rice, and corn, while common 
    weeds include crabgrass, pigweed, and dandelions.
    """
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please ask a valid question."})

    try:
        # Dummy response for the chatbot (replace with a trained QA model)
        if "crop" in question.lower():
            return jsonify({"answer": "Crops are cultivated plants like wheat, rice, or corn."})
        elif "weed" in question.lower():
            return jsonify({"answer": "Weeds are unwanted plants like crabgrass or dandelions."})
        else:
            return jsonify({"answer": "I’m not sure about that. Can you ask a specific question?"})
    except Exception as e:
        return jsonify({"answer": f"An error occurred while processing your question: {e}"})

# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
