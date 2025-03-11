"""
Web scraper for plant disease information.
Handles searching for and fetching content from trusted sources.
"""

import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import sys
import os

from config import TRUSTED_DOMAINS, DEFAULT_HEADERS
import utils


class WebScraper:
    def __init__(self, base_dir: str = "./plant_diseases"):
        """
        Initialize the web scraper.

        Args:
            base_dir: Base directory for caching results
        """
        self.base_dir = base_dir
        self.headers = DEFAULT_HEADERS
        self.trusted_domains = TRUSTED_DOMAINS

        # Ensure the base directory exists
        utils.create_directory_if_not_exists(base_dir)

    def search_google(self, query: str, num_results: int = 10) -> List[str]:
        """
        Search for plant disease information.
        Uses the googlesearch-python library if available, otherwise returns mock data.

        Args:
            query: Search query
            num_results: Number of results to fetch

        Returns:
            List of URLs
        """
        try:
            from googlesearch import search
            results = list(search(query, num_results=num_results))
            print(f"Found {len(results)} search results")
            return results
        except ImportError:
            print("Warning: googlesearch-python not installed. Using mock search results.")
            plant_name = query.split()[0].lower()

            # Dynamic mock results based on plant name
            mock_domains = {
                "tomato": [
                    "https://extension.umn.edu/plant-diseases/tomato-diseases",
                    "https://extension.umd.edu/resource/leaf-spots-tomato",
                    "https://ipm.ucanr.edu/agriculture/tomato/",
                    "https://extension.psu.edu/tomato-diseases"
                ],
                "apple": [
                    "https://extension.umn.edu/plant-diseases/apple-scab",
                    "https://extension.psu.edu/apple-diseases",
                    "https://extension.missouri.edu/publications/g6026",
                    "https://extension.unh.edu/blog/2019/05/common-apple-diseases"
                ],
                "rose": [
                    "https://extension.psu.edu/rose-diseases",
                    "https://extension.umn.edu/plant-diseases/rose-diseases",
                    "https://rhs.org.uk/plants/roses/diseases",
                    "https://gardeningknowhow.com/ornamental/flowers/roses/common-rose-diseases"
                ]
            }

            # Default mock results if plant not in predefined list
            default_results = [
                f"https://extension.umn.edu/plant-diseases/{plant_name}-diseases",
                f"https://extension.psu.edu/{plant_name}-diseases",
                f"https://gardeningknowhow.com/plant-problems/{plant_name}-diseases",
                f"https://rhs.org.uk/plants/{plant_name}/diseases"
            ]

            return mock_domains.get(plant_name, default_results)

    def filter_trusted_urls(self, urls: List[str], max_urls: int = 8) -> List[str]:
        """
        Filter URLs to only include trusted domains.

        Args:
            urls: List of URLs to filter
            max_urls: Maximum number of URLs to return

        Returns:
            List of trusted URLs
        """
        trusted_urls = [url for url in urls if utils.is_domain_trusted(url, self.trusted_domains)]

        if not trusted_urls:
            print("Warning: No trusted domains found in search results. Using top results.")
            trusted_urls = urls[:max_urls]
        else:
            print(f"Found {len(trusted_urls)} results from trusted domains.")

        return trusted_urls[:max_urls]

    def fetch_url(self, url: str, plant_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch content from a URL, using cache if available.

        Args:
            url: URL to fetch
            plant_name: Name of the plant (for caching)

        Returns:
            Dictionary with the fetched content or None if failed
        """
        try:
            # Get the cache directory and filename
            cache_dir = utils.get_cache_directory(self.base_dir, plant_name)
            cache_filename = utils.get_cache_filename(cache_dir, url)

            # Check cache first
            cached_data = utils.load_from_cache(cache_filename)
            if cached_data:
                print(f"Retrieved from cache: {url}")
                return cached_data

            # Fetch the URL
            print(f"Fetching: {url}")
            response = requests.get(url, headers=self.headers, timeout=15)

            if response.status_code != 200:
                print(f"Failed to retrieve content from {url}, status code: {response.status_code}")
                return None

            # Successfully fetched content
            result = {
                "url": url,
                "content": response.text,
                "timestamp": utils.get_timestamp()
            }

            # Cache the result
            utils.save_to_cache(cache_filename, result)

            return result

        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return None

    def search_and_fetch(self, plant_name: str, disease_name: Optional[str] = None,
                         max_results: int = 8) -> List[Dict[str, Any]]:
        """
        Search for plant disease information and fetch content from trusted sources.

        Args:
            plant_name: Name of the plant
            disease_name: Specific disease name (optional)
            max_results: Maximum number of results to fetch

        Returns:
            List of dictionaries with content from trusted sources
        """
        # Construct the search query
        if disease_name:
            query = f"{plant_name} {disease_name} disease symptoms treatment"
        else:
            query = f"{plant_name} common diseases symptoms treatment prevention"

        print(f"Searching for: {query}")

        # Search for URLs
        search_results = self.search_google(query, num_results=15)

        # Filter to trusted domains
        trusted_urls = self.filter_trusted_urls(search_results, max_urls=max_results)

        # Fetch content from each URL
        results = []
        for url in trusted_urls:
            result = self.fetch_url(url, plant_name)
            if result:
                results.append(result)

        print(f"Successfully fetched {len(results)} pages")
        return results


# Simple test function
def test_scraper():
    scraper = WebScraper()
    results = scraper.search_and_fetch("tomato", max_results=2)
    print(f"Fetched {len(results)} results")
    for result in results:
        print(f"URL: {result['url']}")


if __name__ == "__main__":
    test_scraper()