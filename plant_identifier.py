import os
from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# Set your API key
API_KEY = "KJITybfeEdXyZGlssUa1U8n7aeTBwCH4pMQ6KDZiFvJu3oT6Gz"
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the URL for Plant.id API
url = "https://api.plant.id/v2/identify"


@app.route('/')
def index():
    return render_template('upload_page.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if an image is uploaded
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Send the image to Plant.id API
        headers = {
            "Api-Key": API_KEY
        }
        files = {
            "images": open(file_path, "rb")
        }

        response = requests.post(url, headers=headers, files=files)

        if response.status_code == 200:
            result = response.json()

            # Find the plant name with the highest probability
            best_suggestion = max(result['suggestions'], key=lambda x: x['probability'])
            plant_name = best_suggestion['plant_name']
            probability = best_suggestion['probability']

            return render_template('identification_result.html', plant_name=plant_name, probability=probability)
        else:
            return f"Error: {response.status_code}, {response.text}", 400

        files["images"].close()

    return "Something went wrong", 400


if __name__ == '__main__':
    app.run(debug=True)
