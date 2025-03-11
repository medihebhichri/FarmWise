"""
data_collector.py
Module for collecting plant information from multiple sources with multilingual support.
Handles YouTube, websites, and Wikipedia data collection in different languages and dialects.
"""

import os
import json
import re
import time
import urllib.parse
import requests
import logging
from bs4 import BeautifulSoup
import wikipedia
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build

# Import configuration
from config import (
    DATA_DIR, LOGS_DIR, YOUTUBE_API_KEY, LLM_SERVER_URL, COMPLETIONS_ENDPOINT,
    DEFAULT_MODEL, SUPPORTED_LANGUAGES, YOUTUBE_LANGUAGE_MAPPING,
    DIALECT_ADJUSTMENTS, GROWTH_FACTOR_KEYS, DEFAULT_GROWTH_FACTORS,
    TRUSTED_WEBSITES
)

# Set up logging
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "data_collector.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataCollector")

class DataCollector:
    """Collects plant information from multiple sources in various languages."""

    def __init__(self):
        """Initialize the data collector with API connections and directories."""
        self.youtube = None

        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # Initialize YouTube API client
        try:
            self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            logger.info("YouTube API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API client: {e}")

    def _create_data_path(self, plant_name, language_code, dialect=None):
        """
        Create directory structure and file paths for data storage.

        Args:
            plant_name (str): Name of the plant
            language_code (str): Language code (e.g., 'en', 'ar')
            dialect (str, optional): Dialect code (e.g., 'US', 'TN')

        Returns:
            tuple: Directory path and file path
        """
        # Clean plant name for folder naming
        clean_name = plant_name.lower().replace(" ", "_")

        # Create language-specific path with dialect if provided
        if dialect:
            lang_path = f"{language_code}_{dialect}"
        else:
            lang_path = language_code

        # Create full directory path
        dir_path = os.path.join(DATA_DIR, clean_name, lang_path)
        os.makedirs(dir_path, exist_ok=True)

        # Create file path for the collected data
        file_path = os.path.join(dir_path, f"{clean_name}_data.json")

        return dir_path, file_path

    def get_language_specific_query(self, plant_name, language_code, dialect=None):
        """
        Generate language and dialect-specific search queries.

        Args:
            plant_name (str): Name of the plant
            language_code (str): Language code
            dialect (str, optional): Dialect code

        Returns:
            dict: Dictionary with query variations
        """
        # Base query structure by language
        base_queries = {
            "en": [f"how to grow {plant_name}", f"{plant_name} gardening guide",
                   f"{plant_name} care tips", f"growing {plant_name} at home"],

            "ar": [f"كيفية زراعة {plant_name}", f"العناية ب{plant_name}",
                   f"دليل زراعة {plant_name}", f"طريقة زراعة {plant_name}"],

            "fr": [f"comment cultiver {plant_name}", f"jardinage {plant_name}",
                   f"culture de {plant_name}", f"soin des plantes {plant_name}"],

            "es": [f"cómo cultivar {plant_name}", f"guía de jardinería {plant_name}",
                   f"cuidado de {plant_name}", f"cultivar {plant_name} en casa"],

            "de": [f"wie man {plant_name} anbaut", f"gartentipps {plant_name}",
                   f"{plant_name} pflege", f"{plant_name} anbauen"]
        }

        # Default to English if language not supported
        if language_code not in base_queries:
            queries = base_queries["en"]
        else:
            queries = base_queries[language_code]

        # Add dialect-specific adjustments if available
        dialect_key = f"{language_code}_{dialect}" if dialect else None
        if dialect_key and dialect_key in DIALECT_ADJUSTMENTS:
            dialect_info = DIALECT_ADJUSTMENTS[dialect_key]

            # Add dialect-specific query terms
            for term in dialect_info.get("query_terms", []):
                queries.append(f"{term} {plant_name}")

            # Add common phrases
            for phrase in dialect_info.get("common_phrases", []):
                queries.append(f"{phrase} {plant_name}")

        # Special case for Tunisian Arabic
        if language_code == "ar" and dialect == "TN":
            # Add specific Tunisian Arabic queries
            queries.extend([
                f"كيفاش نزرع {plant_name}",  # How to grow (Tunisian)
                f"باش نعتني ب{plant_name}",  # How to care for (Tunisian)
                f"{plant_name} فلاحة تونسية",  # Tunisian agriculture
                f"زراعة {plant_name} في تونس"  # Growing in Tunisia
            ])

        return {
            "queries": queries,
            "language": language_code,
            "dialect": dialect
        }

    # ===========================
    # YouTube Data Collection
    # ===========================

    def search_youtube_videos(self, queries, language_code, max_results=5):
        """
        Search YouTube for videos matching the queries in the specified language.

        Args:
            queries (list): List of search queries
            language_code (str): Language code
            max_results (int): Maximum number of videos to return

        Returns:
            list: List of video information dictionaries
        """
        if not self.youtube:
            logger.error("YouTube API client not initialized. Cannot search videos.")
            return []

        # Map language code for YouTube if necessary
        yt_language = YOUTUBE_LANGUAGE_MAPPING.get(language_code, language_code)

        all_videos = []
        seen_video_ids = set()  # To avoid duplicates

        for query in queries:
            try:
                logger.info(f"Searching YouTube for: '{query}' in {language_code}")

                search_response = self.youtube.search().list(
                    q=query,
                    part='id,snippet',
                    maxResults=max_results,
                    type='video',
                    relevanceLanguage=yt_language
                ).execute()

                for item in search_response.get('items', []):
                    video_id = item['id']['videoId']

                    # Skip if we've already found this video
                    if video_id in seen_video_ids:
                        continue

                    seen_video_ids.add(video_id)
                    title = item['snippet']['title']
                    description = item['snippet']['description']
                    video_url = f"https://www.youtube.com/watch?v={video_id}"

                    all_videos.append({
                        'id': video_id,
                        'title': title,
                        'description': description,
                        'url': video_url,
                        'language': language_code,
                        'source_query': query
                    })

                # Respect YouTube API rate limits
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error searching YouTube for '{query}': {e}")
                continue

        logger.info(f"Found {len(all_videos)} unique YouTube videos across all queries")
        return all_videos[:max_results]  # Limit to max_results total videos

    def fetch_transcript(self, video_id, target_language):
        """
        Fetch the transcript of a YouTube video in the target language if available.

        Args:
            video_id (str): YouTube video ID
            target_language (str): Target language code

        Returns:
            dict: Transcript information including text and language
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try to get transcript in target language
            try:
                transcript = transcript_list.find_transcript([target_language])
                actual_lang = target_language
                logger.info(f"Found transcript in requested language: {target_language}")
            except Exception:
                # If target language not available, try to find any available transcript
                logger.info(f"Target language {target_language} not available, trying alternatives")

                # Try English as fallback
                try:
                    transcript = transcript_list.find_transcript(['en'])
                    actual_lang = 'en'
                    logger.info(f"Using English transcript for video {video_id}")
                except Exception:
                    # Try to get any available transcript
                    available_transcripts = transcript_list.find_generated_transcript()
                    transcript = available_transcripts
                    actual_lang = transcript.language_code
                    logger.info(f"Using {actual_lang} transcript for video {video_id}")

            # Fetch the transcript content
            full_transcript = transcript.fetch()
            transcript_text = " ".join([entry['text'] for entry in full_transcript])

            return {
                "text": transcript_text,
                "language": actual_lang,
                "target_language": target_language,
                "is_translation_needed": actual_lang != target_language
            }

        except Exception as e:
            logger.error(f"Error fetching transcript for video {video_id}: {e}")
            return {
                "text": f"Error: {str(e)}",
                "language": "unknown",
                "target_language": target_language,
                "is_translation_needed": False
            }

    def extract_information_from_transcript(self, transcript_text, plant_name, language_code):
        """
        Extract plant growth information from transcript using LLM.

        Args:
            transcript_text (str): Video transcript text
            plant_name (str): Name of the plant
            language_code (str): Language code

        Returns:
            dict: Extracted information including summary and growth factors
        """
        # Truncate very long transcripts to prevent token limits
        if len(transcript_text) > 8000:
            transcript_text = transcript_text[:8000] + "..."

        # Create language-specific prompt for extraction
        prompt_templates = {
            "en": f"Extract information about growing {plant_name} from this transcript. Provide: 1) A summary of key growing tips, and 2) Specific growth factors including temperature, humidity, soil type, light needs, watering, fertilizer, pH level, spacing. Return as JSON: {{\"summary\": \"...\", \"growth_factors\": {{\"temperature\": \"...\", ...}}}}. Transcript: {transcript_text}",

            "ar": f"استخرج معلومات عن زراعة {plant_name} من هذا النص. قدم: 1) ملخصًا لنصائح الزراعة الرئيسية، و 2) عوامل نمو محددة بما في ذلك درجة الحرارة، والرطوبة، ونوع التربة، واحتياجات الضوء، والري، والأسمدة، ودرجة الحموضة، والتباعد. أعد النتيجة بتنسيق JSON: {{\"summary\": \"...\", \"growth_factors\": {{\"temperature\": \"...\", ...}}}}. النص: {transcript_text}",

            "fr": f"Extrayez des informations sur la culture de {plant_name} à partir de cette transcription. Fournissez: 1) Un résumé des principaux conseils de culture, et 2) Des facteurs de croissance spécifiques, notamment la température, l'humidité, le type de sol, les besoins en lumière, l'arrosage, l'engrais, le pH, l'espacement. Retournez en format JSON: {{\"summary\": \"...\", \"growth_factors\": {{\"temperature\": \"...\", ...}}}}. Transcription: {transcript_text}"
        }

        # Default to English if language not supported
        prompt = prompt_templates.get(language_code, prompt_templates["en"])

        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "max_tokens": 800,
            "temperature": 0.3
        }

        try:
            response = requests.post(COMPLETIONS_ENDPOINT, json=payload)
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    result_text = data['choices'][0].get("text", "").strip()

                    try:
                        # Try to parse JSON response
                        extracted_info = json.loads(result_text)

                        # Ensure growth factors are present and complete
                        if "growth_factors" not in extracted_info:
                            extracted_info["growth_factors"] = DEFAULT_GROWTH_FACTORS.copy()
                        else:
                            # Fill in any missing growth factors
                            for key in GROWTH_FACTOR_KEYS:
                                if key not in extracted_info["growth_factors"]:
                                    extracted_info["growth_factors"][key] = "Not specified"

                        # Ensure summary is present
                        if "summary" not in extracted_info:
                            extracted_info["summary"] = "Summary not available"

                        return extracted_info

                    except json.JSONDecodeError:
                        # Try to extract JSON from text if direct parsing fails
                        match = re.search(r'\{.*\}', result_text, re.DOTALL)
                        if match:
                            try:
                                extracted_info = json.loads(match.group(0))

                                # Fill in missing information
                                if "growth_factors" not in extracted_info:
                                    extracted_info["growth_factors"] = DEFAULT_GROWTH_FACTORS.copy()
                                if "summary" not in extracted_info:
                                    extracted_info["summary"] = "Summary not available"

                                return extracted_info
                            except Exception:
                                pass

                # If we get here, extraction failed
                logger.warning("Failed to extract structured information from transcript")
                return {
                    "summary": "Information extraction failed",
                    "growth_factors": DEFAULT_GROWTH_FACTORS.copy()
                }
            else:
                logger.error(f"LLM server error: {response.status_code}")
                return {
                    "summary": f"LLM server error: {response.status_code}",
                    "growth_factors": DEFAULT_GROWTH_FACTORS.copy()
                }
        except Exception as e:
            logger.error(f"Exception in LLM processing: {e}")
            return {
                "summary": f"Exception during processing: {str(e)}",
                "growth_factors": DEFAULT_GROWTH_FACTORS.copy()
            }

    def collect_youtube_data(self, plant_name, language_code, dialect=None, max_videos=5):
        """
        Collect plant growth information from YouTube videos.

        Args:
            plant_name (str): Name of the plant
            language_code (str): Language code
            dialect (str, optional): Dialect code
            max_videos (int): Maximum number of videos to collect

        Returns:
            list: Collected video data
        """
        # Get language-specific search queries
        query_info = self.get_language_specific_query(plant_name, language_code, dialect)
        queries = query_info["queries"]

        # Search for videos
        videos = self.search_youtube_videos(queries, language_code, max_videos)

        video_data = []
        for video in videos:
            video_id = video["id"]
            logger.info(f"Processing video: {video['title']} (ID: {video_id})")

            # Fetch transcript
            transcript_info = self.fetch_transcript(video_id, language_code)
            transcript_text = transcript_info["text"]

            if transcript_text.startswith("Error"):
                logger.warning(f"Skipping video {video_id} due to transcript error")
                continue

            # Extract information from transcript
            extracted_info = self.extract_information_from_transcript(
                transcript_text, plant_name, transcript_info["language"]
            )

            # Add all information to video data
            video_info = {
                "video_id": video_id,
                "title": video["title"],
                "url": video["url"],
                "description": video["description"],
                "requested_language": language_code,
                "transcript_language": transcript_info["language"],
                "transcript_text": transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text,
                "summary": extracted_info["summary"],
                "growth_factors": extracted_info["growth_factors"]
            }

            video_data.append(video_info)

        logger.info(f"Collected data from {len(video_data)} YouTube videos")
        return video_data

    # ===========================
    # Website Data Collection
    # ===========================

    def get_trusted_websites(self, language_code):
        """
        Get list of trusted websites for a language.

        Args:
            language_code (str): Language code

        Returns:
            list: List of trusted domain names
        """
        return TRUSTED_WEBSITES.get(language_code, TRUSTED_WEBSITES["default"])

    def search_garden_websites(self, plant_name, language_code, dialect=None, max_sites=3):
        """
        Search for information about a plant on gardening websites.

        Args:
            plant_name (str): Name of the plant
            language_code (str): Language code
            dialect (str, optional): Dialect code
            max_sites (int): Maximum number of websites to search

        Returns:
            list: List of relevant URLs
        """
        # Get trusted domains for this language
        trusted_domains = self.get_trusted_websites(language_code)

        # Get language-specific search queries
        query_info = self.get_language_specific_query(plant_name, language_code, dialect)
        queries = query_info["queries"][:2]  # Just use a couple of queries

        relevant_urls = []

        # Search each trusted domain with each query
        for domain in trusted_domains[:max_sites]:
            for query in queries:
                search_url = f"https://www.google.com/search?q=site:{domain}+{urllib.parse.quote(query)}"

                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    response = requests.get(search_url, headers=headers)

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # Extract URLs from search results
                        for link in soup.find_all('a'):
                            href = link.get('href', '')

                            if "/url?q=" in href and domain in href:
                                url = href.split("/url?q=")[1].split("&")[0]

                                # Only add unique URLs
                                if url not in relevant_urls:
                                    relevant_urls.append(url)
                                    break  # Just take the first result per query-domain pair

                    # Respect rate limits
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Error searching {domain} with query '{query}': {e}")

        logger.info(f"Found {len(relevant_urls)} relevant garden website URLs")
        return relevant_urls

    def scrape_website_content(self, url):
        """
        Scrape content from a gardening website.

        Args:
            url (str): Website URL

        Returns:
            dict: Scraped content information
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove non-content elements
                for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'meta', 'iframe']):
                    tag.decompose()

                # Try to detect the language
                html_lang = soup.html.get('lang', '') if soup.html else ''
                detected_language = html_lang[:2] if html_lang else 'unknown'

                # Extract the page title
                title = soup.title.string if soup.title else url

                # Look for main content container
                content_text = ""

                # Try to find the main content area
                main_content = soup.find(['article', 'main', 'div.content', 'div.post-content', 'div.entry-content'])

                if main_content:
                    # Extract text from main content area
                    for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
                        text = element.get_text().strip()
                        if text and len(text) > 15:  # Skip very short fragments
                            content_text += text + "\n\n"
                else:
                    # If no main content area found, extract from body
                    for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
                        text = element.get_text().strip()
                        if text and len(text) > 15:
                            content_text += text + "\n\n"

                return {
                    "url": url,
                    "domain": urllib.parse.urlparse(url).netloc,
                    "title": title,
                    "content": content_text,
                    "detected_language": detected_language,
                    "status": "success"
                }
            else:
                return {
                    "url": url,
                    "domain": urllib.parse.urlparse(url).netloc,
                    "status": "error",
                    "error": f"HTTP Status {response.status_code}"
                }
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {
                "url": url,
                "domain": urllib.parse.urlparse(url).netloc if url.startswith('http') else "unknown",
                "status": "error",
                "error": str(e)
            }

    def extract_info_from_web_content(self, content, plant_name, language_code):
        """
        Extract plant growth information from web content using LLM.

        Args:
            content (str): Website content
            plant_name (str): Name of the plant
            language_code (str): Language code

        Returns:
            dict: Extracted information including summary and growth factors
        """
        # Truncate very long content to prevent token limits
        if len(content) > 8000:
            content = content[:8000] + "..."

        # Create language-specific prompt for extraction
        prompt_templates = {
            "en": f"Extract information about growing {plant_name} from this gardening website content. Provide: 1) A summary of key growing tips, and 2) Specific growth factors including temperature, humidity, soil type, light needs, watering, fertilizer, pH level, spacing. Return as JSON: {{\"summary\": \"...\", \"growth_factors\": {{\"temperature\": \"...\", ...}}}}. Content: {content}",

            "ar": f"استخرج معلومات عن زراعة {plant_name} من محتوى موقع البستنة هذا. قدم: 1) ملخصًا لنصائح الزراعة الرئيسية، و 2) عوامل نمو محددة بما في ذلك درجة الحرارة، والرطوبة، ونوع التربة، واحتياجات الضوء، والري، والأسمدة، ودرجة الحموضة، والتباعد. أعد النتيجة بتنسيق JSON: {{\"summary\": \"...\", \"growth_factors\": {{\"temperature\": \"...\", ...}}}}. المحتوى: {content}",

            "fr": f"Extrayez des informations sur la culture de {plant_name} à partir de ce contenu de site web de jardinage. Fournissez: 1) Un résumé des principaux conseils de culture, et 2) Des facteurs de croissance spécifiques, notamment la température, l'humidité, le type de sol, les besoins en lumière, l'arrosage, l'engrais, le pH, l'espacement. Retournez en format JSON: {{\"summary\": \"...\", \"growth_factors\": {{\"temperature\": \"...\", ...}}}}. Contenu: {content}"
        }

        # Default to English if language not supported
        prompt = prompt_templates.get(language_code, prompt_templates["en"])

        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "max_tokens": 800,
            "temperature": 0.3
        }

        try:
            response = requests.post(COMPLETIONS_ENDPOINT, json=payload)
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    result_text = data['choices'][0].get("text", "").strip()

                    try:
                        # Try to parse JSON response
                        extracted_info = json.loads(result_text)

                        # Ensure growth factors are present and complete
                        if "growth_factors" not in extracted_info:
                            extracted_info["growth_factors"] = DEFAULT_GROWTH_FACTORS.copy()
                        else:
                            # Fill in any missing growth factors
                            for key in GROWTH_FACTOR_KEYS:
                                if key not in extracted_info["growth_factors"]:
                                    extracted_info["growth_factors"][key] = "Not specified"

                        # Ensure summary is present
                        if "summary" not in extracted_info:
                            extracted_info["summary"] = "Summary not available"

                        return extracted_info

                    except json.JSONDecodeError:
                        # Try to extract JSON from text if direct parsing fails
                        match = re.search(r'\{.*\}', result_text, re.DOTALL)
                        if match:
                            try:
                                extracted_info = json.loads(match.group(0))

                                # Fill in missing information
                                if "growth_factors" not in extracted_info:
                                    extracted_info["growth_factors"] = DEFAULT_GROWTH_FACTORS.copy()
                                if "summary" not in extracted_info:
                                    extracted_info["summary"] = "Summary not available"

                                return extracted_info
                            except Exception:
                                pass

                # If we get here, extraction failed
                logger.warning("Failed to extract structured information from web content")
                return {
                    "summary": "Information extraction failed",
                    "growth_factors": DEFAULT_GROWTH_FACTORS.copy()
                }
            else:
                logger.error(f"LLM server error: {response.status_code}")
                return {
                    "summary": f"LLM server error: {response.status_code}",
                    "growth_factors": DEFAULT_GROWTH_FACTORS.copy()
                }
        except Exception as e:
            logger.error(f"Exception in LLM processing: {e}")
            return {
                "summary": f"Exception during processing: {str(e)}",
                "growth_factors": DEFAULT_GROWTH_FACTORS.copy()
            }

    def collect_web_data(self, plant_name, language_code, dialect=None, max_sites=3):
        """
        Collect plant growth information from gardening websites.

        Args:
            plant_name (str): Name of the plant
            language_code (str): Language code
            dialect (str, optional): Dialect code
            max_sites (int): Maximum number of websites to collect

        Returns:
            list: Collected website data
        """
        logger.info(f"Searching gardening websites for {plant_name} in {language_code}")

        # Get relevant URLs
        urls = self.search_garden_websites(plant_name, language_code, dialect, max_sites)

        web_data = []
        for url in urls:
            logger.info(f"Scraping content from: {url}")

            # Scrape content
            content_info = self.scrape_website_content(url)

            if content_info["status"] == "error":
                logger.warning(f"Skipping {url} due to scraping error: {content_info.get('error')}")
                continue

            # Extract information from content
            content_text = content_info["content"]
            extracted_info = self.extract_info_from_web_content(content_text, plant_name, language_code)

            # Add all information to website data
            site_info = {
                "url": url,
                "domain": content_info["domain"],
                "title": content_info["title"],
                "detected_language": content_info["detected_language"],
                "content_sample": content_text[:500] + "..." if len(content_text) > 500 else content_text,
                "summary": extracted_info["summary"],
                "growth_factors": extracted_info["growth_factors"]
            }

            web_data.append(site_info)

        logger.info(f"Collected data from {len(web_data)} websites")
        return web_data

    # ===========================
    # Wikipedia Data Collection
    # ===========================

    def set_wikipedia_language(self, language_code):
        """
        Set Wikipedia language for searches.

        Args:
            language_code (str): Language code
        """
        # Map language codes to Wikipedia language codes if needed
        wiki_lang_map = {
            "en": "en",
            "ar": "ar",
            "fr": "fr",
            "es": "es",
            "de": "de",
            "it": "it",
            "pt": "pt",
            "ru": "ru",
            "zh": "zh",
            "ja": "ja",
            "hi": "hi",
            "tr": "tr"
        }

        # Default to English if language not supported
        wiki_lang = wiki_lang_map.get(language_code, "en")

        # Set Wikipedia language
        wikipedia.set_lang(wiki_lang)
        logger.info(f"Set Wikipedia language to: {wiki_lang}")

    def get_wikipedia_data(self, plant_name, language_code):
        """
        Get information about a plant from Wikipedia.

        Args:
            plant_name (str): Name of the plant
            language_code (str): Language code

        Returns:
            dict: Wikipedia data
        """
        logger.info(f"Getting Wikipedia data for {plant_name} in {language_code}")

        # Set Wikipedia language
        self.set_wikipedia_language(language_code)

        try:
            # Try different search queries
            search_queries = [
                f"{plant_name}",
                f"{plant_name} plant",
                f"{plant_name} tree" if len(plant_name.split()) == 1 else plant_name,
                f"{plant_name} flower" if len(plant_name.split()) == 1 else plant_name
            ]

            # Try each search query
            for query in search_queries:
                try:
                    # Search for the plant
                    search_results = wikipedia.search(query, results=3)

                    if not search_results:
                        continue

                    # Try each search result
                    for result in search_results:
                        try:
                            # Get page content
                            wiki_page = wikipedia.page(result)
                            title = wiki_page.title
                            url = wiki_page.url
                            content = wiki_page.content

                            # Try to find cultivation/growing sections
                            relevant_section_keywords = [
                                "Cultivation", "Growing", "Care", "Gardening",
                                "Horticulture", "Agriculture", "Planting",
                                "Culture", "Farming", "Uses", "Propagation"
                            ]

                            sections = wiki_page.sections
                            section_content = ""

                            for section in sections:
                                if any(keyword.lower() in section.lower() for keyword in relevant_section_keywords):
                                    try:
                                        section_text = wiki_page.section(section)
                                        if section_text:
                                            section_content += f"{section}:\n{section_text}\n\n"
                                    except Exception:
                                        continue

                            # If no relevant sections found, use summary
                            if not section_content:
                                section_content = wikipedia.summary(result, sentences=10)

                            # Extract growth factors
                            extracted_info = self.extract_info_from_web_content(
                                section_content, plant_name, language_code
                            )

                            return {
                                "title": title,
                                "url": url,
                                "content": section_content,
                                "language": language_code,
                                "summary": extracted_info["summary"],
                                "growth_factors": extracted_info["growth_factors"],
                                "source": "wikipedia"
                            }

                        except wikipedia.exceptions.DisambiguationError as e:
                            # Try the first option
                            try:
                                new_title = e.options[0]
                                wiki_summary = wikipedia.summary(new_title, sentences=10)

                                # Extract growth factors
                                extracted_info = self.extract_info_from_web_content(
                                    wiki_summary, plant_name, language_code
                                )

                                return {
                                    "title": new_title,
                                    "url": f"https://{language_code}.wikipedia.org/wiki/{urllib.parse.quote(new_title)}",
                                    "content": wiki_summary,
                                    "language": language_code,
                                    "summary": extracted_info["summary"],
                                    "growth_factors": extracted_info["growth_factors"],
                                    "source": "wikipedia"
                                }
                            except Exception:
                                continue
                        except Exception as e:
                            logger.error(f"Error with Wikipedia result '{result}': {e}")
                            continue

                except Exception as e:
                    logger.error(f"Error with Wikipedia search query '{query}': {e}")
                    continue

            logger.warning(f"No Wikipedia data found for {plant_name} in {language_code}")
            return {
                "title": f"{plant_name} (not found)",
                "url": "",
                "content": "",
                "language": language_code,
                "summary": f"No Wikipedia information found for {plant_name}.",
                "growth_factors": DEFAULT_GROWTH_FACTORS.copy(),
                "source": "wikipedia"
            }

        except Exception as e:
            logger.error(f"Error getting Wikipedia content: {e}")
            return {
                "title": f"{plant_name} (error)",
                "url": "",
                "content": "",
                "language": language_code,
                "summary": f"Error retrieving Wikipedia content: {e}",
                "growth_factors": DEFAULT_GROWTH_FACTORS.copy(),
                "source": "wikipedia"
            }

    # ===========================
    # Combined Data Collection
    # ===========================

    def collect_plant_data(self, plant_name, language_code, dialect=None, max_videos=3, max_websites=3):
        """
        Collect all available data for a plant and save to disk.

        Args:
            plant_name (str): Name of the plant
            language_code (str): Language code
            dialect (str, optional): Dialect code
            max_videos (int): Maximum number of videos to process
            max_websites (int): Maximum number of websites to process

        Returns:
            dict: All collected data
        """
        logger.info(f"Collecting data for {plant_name} in {language_code}" +
                    (f" ({dialect} dialect)" if dialect else ""))

        # Create directories and file paths
        dir_path, file_path = self._create_data_path(plant_name, language_code, dialect)

        # Check if we already have data for this plant in this language
        if os.path.exists(file_path):
            logger.info(f"Data already exists at {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                # Return existing data if it's complete
                if all(k in existing_data for k in ["youtube_data", "web_data", "wikipedia_data"]):
                    logger.info(f"Returning existing data for {plant_name}")
                    return existing_data
            except Exception:
                logger.warning(f"Could not read existing data file, will collect new data")

        # Collect data from different sources
        youtube_data = self.collect_youtube_data(plant_name, language_code, dialect, max_videos)
        web_data = self.collect_web_data(plant_name, language_code, dialect, max_websites)
        wikipedia_data = self.get_wikipedia_data(plant_name, language_code)

        # Combine all data
        plant_data = {
            "plant_name": plant_name,
            "language": language_code,
            "dialect": dialect,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "youtube_data": youtube_data,
            "web_data": web_data,
            "wikipedia_data": wikipedia_data
        }

        # Save combined data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(plant_data, f, indent=2, ensure_ascii=False)

        logger.info(f"All data for {plant_name} saved to {file_path}")
        return plant_data

    def collect_data_for_multiple_plants(self, plant_list, language_code, dialect=None, max_videos=3, max_websites=2):
        """
        Collect data for multiple plants.

        Args:
            plant_list (list): List of plant names
            language_code (str): Language code
            dialect (str, optional): Dialect code
            max_videos (int): Maximum videos per plant
            max_websites (int): Maximum websites per plant

        Returns:
            dict: Dictionary with data for each plant
        """
        all_plants_data = {}

        for plant_name in plant_list:
            if not plant_name.strip():
                continue

            plant_data = self.collect_plant_data(
                plant_name.strip(),
                language_code,
                dialect,
                max_videos,
                max_websites
            )

            all_plants_data[plant_name] = plant_data

        return all_plants_data

# For testing
if __name__ == "__main__":
    collector = DataCollector()

    # Test collecting data for a plant in different languages
    test_plant = input("Enter a plant name to test: ")
    test_language = input("Enter language code (e.g., en, ar, fr): ")
    test_dialect = input("Enter dialect code (optional, e.g., US, TN): ") or None

    collector.collect_plant_data(
        plant_name=test_plant,
        language_code=test_language,
        dialect=test_dialect,
        max_videos=2,
        max_websites=2
    )