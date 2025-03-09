import csv
import json
import os
import re
import urllib.parse
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build

# ===========================
# Configuration Section
# ===========================
# YouTube API configuration
API_KEY = 'AIzaSyA6IIgizjfr7O66rI-KeDOdjlm_BkdvFV4'

# LM Studio configuration
LM_SERVER_URL = "http://localhost:1234"  # Update if necessary
MODELS_ENDPOINT = f"{LM_SERVER_URL}/v1/models"
COMPLETIONS_ENDPOINT = f"{LM_SERVER_URL}/v1/completions"
DEFAULT_MODEL = "your-model"  # e.g., "llama-2-7b"

# Define the fixed keys for growth factors
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


def _extract_optimal_factors(self, plant_data):
    """
    Extract the optimal growth factors from a plant's data.
    This uses a simple approach of taking the most common values across videos.
    """
    # Use the global GROWTH_FACTOR_KEYS variable
    factor_values = {factor: [] for factor in GROWTH_FACTOR_KEYS}

    # Collect all values for each factor
    for video in plant_data.get("videos", []):
        growth_factors = video.get("growth_factors", {})

        for factor in GROWTH_FACTOR_KEYS:
            value = growth_factors.get(factor, "Not specified")
            if value != "Not specified":
                # Convert non-scalar types to JSON strings
                if isinstance(value, (dict, list)):
                    processed_value = json.dumps(value, sort_keys=True)
                else:
                    processed_value = value
                factor_values[factor].append(processed_value)

    # Determine the optimal value for each factor
    optimal_factors = {}
    for factor, values in factor_values.items():
        if values:
            # Take the most common value as optimal
            optimal_factors[factor] = max(set(values), key=values.count)
        else:
            optimal_factors[factor] = "Not specified"

    return optimal_factors

def search_videos(query, max_results=5):
    """
    Searches YouTube for videos based on the query.
    """
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=max_results,
        type='video'
    ).execute()

    video_urls = []
    for item in search_response['items']:
        video_id = item['id']['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_urls.append(video_url)

    return video_urls


def fetch_transcript(video_url):
    """
    Fetches the transcript of a YouTube video.
    """
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            return f"Could not extract video ID from URL: {video_url}"

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en'])
        full_transcript = transcript.fetch()
        transcript_text = " ".join([entry['text'] for entry in full_transcript])
        return transcript_text
    except Exception as e:
        return str(e)


def create_folder_structure(plant_name):
    """
    Creates the folder structure for a plant.
    Returns the paths to the CSV and JSON folders.
    """
    # Clean plant name for folder name
    clean_name = plant_name.lower().replace(" ", "_")

    # Create main plant folder
    plant_folder = os.path.join(os.getcwd(), clean_name)
    os.makedirs(plant_folder, exist_ok=True)

    # Create CSV and JSON subfolders
    csv_folder = os.path.join(plant_folder, "transcripts")
    json_folder = os.path.join(plant_folder, "summaries")

    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    return csv_folder, json_folder


def process_plant_query(plant_query, max_results=5):
    """
    Processes a query for a plant by:
    1. Searching for videos
    2. Fetching transcripts
    3. Creating folder structure
    4. Saving transcripts to CSV
    5. Processing transcripts with LM Studio
    6. Saving all summaries to a single JSON file
    """
    print(f"\n{'=' * 50}")
    print(f"Processing plant: {plant_query}")
    print(f"{'=' * 50}")

    # Extract the plant name from the query
    plant_name = plant_query.split("for ")[-1].strip()
    if not plant_name:
        plant_name = plant_query

    # Create folder structure
    csv_folder, json_folder = create_folder_structure(plant_name)

    # Search for videos
    print(f"Searching for videos about {plant_name}...")
    video_urls = search_videos(plant_query, max_results)

    if not video_urls:
        print(f"No videos found for query: {plant_query}")
        return

    # Prepare CSV file
    csv_file_path = os.path.join(csv_folder, f"{plant_name.lower().replace(' ', '_')}_transcripts.csv")

    # Fetch and save transcripts
    all_video_data = []

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Video URL', 'Video ID', 'Transcript'])

        for video_url in video_urls:
            video_id = extract_video_id(video_url)
            if not video_id:
                print(f"Could not extract video ID from URL: {video_url}")
                continue

            print(f"Fetching transcript for: {video_url}")
            transcript = fetch_transcript(video_url)

            if "Error" in transcript or "Exception" in transcript:
                print(f"Error fetching transcript: {transcript}")
                csvwriter.writerow([video_url, video_id, f"Error: {transcript}"])
                continue

            # Write to CSV
            csvwriter.writerow([video_url, video_id, transcript])

            # Process with LM Studio if it's available
            if check_lm_studio_status():
                print(f"Summarizing video {video_id}...")
                summary = summarize_text(transcript)

                print(f"Extracting growth factors for video {video_id}...")
                growth_factors = extract_growth_factors(transcript)

                # Add to the list of video data
                video_data = {
                    "video_id": video_id,
                    "video_url": video_url,
                    "summary": summary,
                    "growth_factors": growth_factors
                }
                all_video_data.append(video_data)
            else:
                print("LM Studio is not available. Skipping summarization and factor extraction.")

    print(f"Saved transcripts to {csv_file_path}")

    # Save all video data to a single JSON file
    if all_video_data:
        json_file_path = os.path.join(json_folder, f"{plant_name.lower().replace(' ', '_')}_summaries.json")
        with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
            plant_summary = {
                "plant_name": plant_name,
                "query": plant_query,
                "videos": all_video_data
            }
            json.dump(plant_summary, jsonfile, indent=4, ensure_ascii=False)
        print(f"Saved all summaries to {json_file_path}")


def main():
    print("Plant Growth Factors Video Organizer")
    print("This program will search for videos about plant growth factors,")
    print("download transcripts, and analyze them with LM Studio.")

    # Get queries or plant names from the user
    input_type = input("Enter '1' to use a text file with plant queries or '2' to enter plant names directly: ").strip()

    plant_queries = []

    if input_type == '1':
        file_path = input("Enter the path to the queries file (comma-separated): ").strip()
        try:
            with open(file_path, 'r') as file:
                content = file.read().strip()
                plant_queries = [query.strip() for query in content.split(',')]
        except FileNotFoundError:
            print("The specified file was not found.")
            return
    elif input_type == '2':
        plants_input = input("Enter plant names separated by commas (e.g., tomato, potato): ").strip()
        plant_names = [name.strip() for name in plants_input.split(',')]

        # Convert plant names to queries
        for plant in plant_names:
            plant_queries.append(f"best factors for growth of {plant}")
    else:
        print("Invalid option selected.")
        return

    # Get number of results per plant
    max_results = int(input("Enter the maximum number of videos to process for each plant (1-20): ") or "5")
    max_results = min(max(1, max_results), 20)  # Ensure it's between 1 and 20

    # Process each plant query
    for query in plant_queries:
        if not query.strip():
            continue
        process_plant_query(query, max_results)

    print("\nProcessing complete!")


if __name__ == '__main__':
    main()