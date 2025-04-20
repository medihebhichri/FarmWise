import requests

# Set your API key
API_KEY = "KJITybfeEdXyZGlssUa1U8n7aeTBwCH4pMQ6KDZiFvJu3oT6Gz"

# Define the URL for Plant.id API
url = "https://api.plant.id/v2/identify"

# Path to your image (use raw string to avoid unicode error)
image_path = r"C:\Users\msi\Desktop\FINAL\images.jpg"

# Read the image and prepare the payload
headers = {
    "Api-Key": API_KEY
}
files = {
    "images": open(image_path, "rb")
}

# Send the request
response = requests.post(url, headers=headers, files=files)

# Check if the response is successful
if response.status_code == 200:
    result = response.json()
    print("Identification Result:", result)
else:
    print(f"Error: {response.status_code}, {response.text}")

# Close the image file
files["images"].close()
