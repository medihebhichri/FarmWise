import csv
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build

API_KEY = 'AIzaSyA6IIgizjfr7O66rI-KeDOdjlm_BkdvFV4'
youtube = build('youtube', 'v3', developerKey=API_KEY)

def search_videos(query, max_results=5):
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
    try:
        video_id = video_url.split("?v=")[1]
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en'])
        full_transcript = transcript.fetch()
        transcript_text = " ".join([entry['text'] for entry in full_transcript])
        return transcript_text
    except Exception as e:
        return str(e)

def main():

    file_path = input("Enter the path to the queries file: ").strip()
    max_results = int(input("Enter the maximum number of results for each query: "))

    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            queries = [query.strip() for query in content.split(',')]
    except FileNotFoundError:
        print("The specified file was not found.")
        return

    for query in queries:
        if not query:
            continue

        print(f"\nSearching for videos related to: {query}")
        video_urls = search_videos(query, max_results)
        rows = []

        for video_url in video_urls:
            print(f"\nFetching transcript for: {video_url}")
            transcript = fetch_transcript(video_url)
            rows.append([video_url, transcript])

            if isinstance(transcript, str):
                print(transcript)
            else:
                print("Error fetching transcript.")

        filename = f'video_transcripts_{query.replace(" ", "_")}.csv'
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Video URL', 'Transcript'])
            csvwriter.writerows(rows)

        print(f"Results for query '{query}' saved to {filename}")

if __name__ == '__main__':
    main()
