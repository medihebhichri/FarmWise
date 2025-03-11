import requests
import time
import random
import uuid


class PixverseAPI:
    """
    A client for interacting with the Pixverse Platform API.
    Updated based on the official documentation.
    """

    def __init__(self, api_key):
        """
        Initialize the Pixverse API client.

        Args:
            api_key (str): Your Pixverse API key
        """
        self.api_key = api_key
        self.base_url = "https://app-api.pixverse.ai"

    def generate_text_to_video(self, prompt, negative_prompt="", aspect_ratio="16:9",
                               duration=5, model="v3.5", motion_mode="normal",
                               quality="540p", seed=0, water_mark=False):
        """
        Generate a video from a text prompt.

        Args:
            prompt (str): The text prompt to generate a video from
            negative_prompt (str): Text to avoid in the generation
            aspect_ratio (str): Aspect ratio of the video (default: "16:9")
            duration (int): Duration in seconds (default: 5)
            model (str): Model version to use (default: "v3.5")
            motion_mode (str): Motion mode for the video (default: "normal")
            quality (str): Video quality (default: "540p")
            seed (int): Random seed for reproducibility (default: 0)
            water_mark (bool): Whether to include a watermark (default: False)

        Returns:
            dict: Response containing the video_id
        """
        endpoint = f"{self.base_url}/openapi/v2/video/text/generate"

        # Generate a unique trace ID for this request
        trace_id = str(uuid.uuid4())

        # Set up headers according to documentation
        headers = {
            "Api-Key": self.api_key,
            "Ai-Trace-Id": trace_id,
            "Content-Type": "application/json"
        }

        # Prepare the payload according to documentation
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "model": model,
            "motion_mode": motion_mode,
            "quality": quality,
            "seed": seed if seed != 0 else random.randint(1, 1000000),
            "water_mark": water_mark
        }

        print(f"Sending request to: {endpoint}")
        print(f"Headers: {headers}")
        print(f"Request payload: {payload}")

        # Send the request to generate the video
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )

            print(f"Response status: {response.status_code}")

            # Raise an exception for 4XX and 5XX responses
            response.raise_for_status()

            # Parse the response as JSON
            result = response.json()
            print(f"Response: {result}")

            # Check for errors in the response
            if result.get("ErrCode", -1) != 0:
                error_message = result.get("ErrMsg", "Unknown error")
                raise Exception(f"API Error: {error_message}")

            # Extract and return the video_id
            video_id = result.get("Resp", {}).get("video_id")
            if not video_id:
                raise Exception("No video_id returned in the response")

            return video_id

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            raise

    def check_video_status(self, video_id):
        """
        Check the status of a video generation.

        Args:
            video_id (int): The ID of the video generation to check

        Returns:
            dict: The video status information
        """
        endpoint = f"{self.base_url}/openapi/v2/video/result/{video_id}"

        headers = {
            "Api-Key": self.api_key
        }

        try:
            response = requests.get(
                endpoint,
                headers=headers,
                timeout=30
            )

            # Raise an exception for 4XX and 5XX responses
            response.raise_for_status()

            # Parse the response as JSON
            result = response.json()

            # Check for errors in the response
            if result.get("ErrCode", -1) != 0:
                error_message = result.get("ErrMsg", "Unknown error")
                raise Exception(f"API Error: {error_message}")

            return result.get("Resp", {})

        except requests.exceptions.RequestException as e:
            print(f"Error checking video status: {e}")
            raise

    def wait_for_video_completion(self, video_id, polling_interval=10, timeout=300):
        """
        Poll until a video generation is complete or fails.

        Args:
            video_id (int): The ID of the video to wait for
            polling_interval (int): How often to check status in seconds
            timeout (int): Maximum time to wait in seconds

        Returns:
            dict: The final video status with URL
        """
        start_time = time.time()
        print(f"Waiting for video {video_id} to complete...")

        while time.time() - start_time < timeout:
            try:
                status_info = self.check_video_status(video_id)
                status_code = status_info.get("status")

                print(f"Current status: {status_code}")

                # Status code 1 means the video is complete according to docs
                if status_code == 1:
                    video_url = status_info.get("url")
                    if not video_url:
                        raise Exception("Video completed but no URL was provided")

                    print(f"Video generation completed!")
                    return status_info

                # If status is 0, it's still processing
                elif status_code == 0:
                    print(f"Video is still processing. Waiting {polling_interval} seconds...")
                # Any other status may indicate failure
                else:
                    print(f"Unexpected status code: {status_code}")
                    print(f"Full status info: {status_info}")

                time.sleep(polling_interval)

            except Exception as e:
                print(f"Error checking status: {e}")
                time.sleep(polling_interval)

        raise TimeoutError(f"Video generation timed out after {timeout} seconds")


# Example usage
def main():
    # Your API key
    api_key = "sk-b0157dc584b2ee6be197a3893732af79"

    # Create the API client
    client = PixverseAPI(api_key)

    # Text prompt for the video
    prompt = "A beautiful sunset over a calm ocean with sailboats on the horizon"

    try:
        # Start the video generation
        print(f"Generating video for prompt: '{prompt}'")

        video_id = client.generate_text_to_video(
            prompt=prompt,
            negative_prompt="poor quality, blurry, distorted",
            aspect_ratio="16:9",
            duration=5,
            model="v3.5",
            motion_mode="normal",
            quality="540p"
        )

        print(f"Video generation started with ID: {video_id}")

        # Wait for the video to complete
        result = client.wait_for_video_completion(video_id)

        # Get the video URL
        video_url = result.get("url")
        print(f"Video URL: {video_url}")

        # Download the video
        if video_url:
            print(f"Downloading video from: {video_url}")
            import urllib.request
            urllib.request.urlretrieve(video_url, 'generated_video.mp4')
            print("Video downloaded successfully as 'generated_video.mp4'")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()