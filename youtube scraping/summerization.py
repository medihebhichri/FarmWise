import csv
import json
import os
import re
import urllib.parse
import requests

# ===========================
# Configuration Section
# ===========================
LM_SERVER_URL = "http://localhost:1234"  # Update if necessary
MODELS_ENDPOINT = f"{LM_SERVER_URL}/v1/models"            # For checking LM Studio status
COMPLETIONS_ENDPOINT = f"{LM_SERVER_URL}/v1/completions"  # For generating summaries and extracting factors

# Set the model name to one that is loaded in LM Studio
DEFAULT_MODEL = "your-model"  # e.g., "llama-2-7b"

# Define the fixed keys for growth factors.
GROWTH_FACTOR_KEYS = [
    "temperature",
    "humidity",
    "method_use",
    "soil_natural",
    "method_use_for_plant",
    "light_exposure",
    "water_requirements",
    "fertilizer",
    "pH_level",
    "plant_spacing"
]

DEFAULT_GROWTH_FACTORS = {key: "Not specified" for key in GROWTH_FACTOR_KEYS}

# ===========================
# Utility Functions
# ===========================

def check_lm_studio_status():
    """
    Checks LM Studio server availability using GET /v1/models.
    """
    try:
        response = requests.get(MODELS_ENDPOINT)
        if response.status_code == 200:
            print("LM Studio is up and running.")
            return True
        else:
            print(f"LM Studio status check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print("Error checking LM Studio status:", e)
        return False

def extract_video_id(video_url):
    """
    Extracts the YouTube video ID from the URL.
    """
    parsed = urllib.parse.urlparse(video_url)
    query = urllib.parse.parse_qs(parsed.query)
    if 'v' in query:
        return query['v'][0]
    # Attempt regex extraction for non-standard URL formats
    pattern = re.compile(r"(?<=/)([a-zA-Z0-9_-]{11})(?=[/?])")
    match = pattern.search(video_url)
    if match:
        return match.group(0)
    return None

def summarize_text(transcript_text):
    """
    Sends the transcript to LM Studio using POST /v1/completions for summarization.
    """
    prompt = f"Summarize the following text:\n{transcript_text}\n\nSummary:"
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "max_tokens": 200  # Adjust as needed for summary length
    }
    try:
        response = requests.post(COMPLETIONS_ENDPOINT, json=payload)
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                summary = data['choices'][0].get("text", "No summary provided").strip()
                return summary
            else:
                return "Summarization failed: No choices in response."
        else:
            return f"Error: LM Studio responded with status {response.status_code}"
    except Exception as e:
        return f"Exception during summarization: {e}"

def extract_growth_factors(transcript_text):
    """
    Uses LM Studio to extract growth factors from the transcript text.
    The prompt instructs the model to return ONLY a valid JSON object with fixed keys.
    If extra text is returned, the function will attempt to extract the JSON substring,
    then filter it to keep only the fixed keys. If extraction fails, default values are used.
    """
    prompt = (
        "Extract the following growth factors from the transcript text. Return ONLY a valid JSON object with exactly these keys: "
        "temperature, humidity, method_use, soil_natural, method_use_for_plant, light_exposure, water_requirements, fertilizer, pH_level, plant_spacing. "
        "For any key not clearly mentioned in the transcript, set its value to a close approximate or 'Not specified'. "
        "Do not include any extra text or commentary. \n\n"
        f"Transcript text:\n{transcript_text}\n\nReturn JSON:"
    )
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "max_tokens": 300  # Adjust if necessary
    }
    try:
        response = requests.post(COMPLETIONS_ENDPOINT, json=payload)
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                result_text = data['choices'][0].get("text", "").strip()
                try:
                    # Try to load the result directly
                    growth_factors = json.loads(result_text)
                except json.JSONDecodeError:
                    # Attempt to extract a JSON substring from the result_text using regex
                    match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if match:
                        try:
                            growth_factors = json.loads(match.group(0))
                        except Exception as e:
                            print("Error decoding extracted JSON:", e)
                            return DEFAULT_GROWTH_FACTORS.copy()
                    else:
                        print("Error decoding growth factors JSON. Using default values.")
                        return DEFAULT_GROWTH_FACTORS.copy()
                # Filter out extra keys and ensure all fixed keys exist
                final_growth_factors = {key: growth_factors.get(key, "Not specified") for key in GROWTH_FACTOR_KEYS}
                return final_growth_factors
            else:
                return DEFAULT_GROWTH_FACTORS.copy()
        else:
            return {key: f"Error: status {response.status_code}" for key in GROWTH_FACTOR_KEYS}
    except Exception as e:
        return {key: f"Exception: {e}" for key in GROWTH_FACTOR_KEYS}

def process_csv_file(csv_file):
    """
    Reads the CSV file containing video transcripts and processes each row.
    Each row should have 'Video URL' and 'Transcript' columns.
    For each video, the script:
      - Extracts the video ID.
      - Summarizes the transcript.
      - Extracts growth factors using LM Studio.
      - Generates a JSON file with a consistent structure.
    """
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_url = row.get("Video URL")
            transcript = row.get("Transcript")
            if not video_url or not transcript:
                print("Skipping a row due to missing Video URL or Transcript.")
                continue

            video_id = extract_video_id(video_url)
            if not video_id:
                print(f"Could not extract video ID from URL: {video_url}")
                continue

            print(f"\nProcessing video: {video_id}")
            summary = summarize_text(transcript)
            growth_factors = extract_growth_factors(transcript)

            # Build the JSON object with a consistent structure
            video_data = {
                "video_id": video_id,
                "summary": summary,
                "growth_factors": growth_factors
            }

            output_file = f"{video_id}.json"
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(video_data, jsonfile, indent=4, ensure_ascii=False)
            print(f"Saved summarized data for video {video_id} to {output_file}")

def main():
    if not check_lm_studio_status():
        print("LM Studio is not available. Exiting.")
        return

    csv_path = input("Enter the path to the CSV file containing video transcripts: ").strip()
    if not os.path.isfile(csv_path):
        print("CSV file not found. Please check the path and try again.")
        return

    process_csv_file(csv_path)

if __name__ == '__main__':
    main()
