"""
Content extractor for processing HTML content from web pages.
Extracts and cleans text content from scraped web pages.
"""

from bs4 import BeautifulSoup
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from config import HTML_CONTENT_SELECTORS
import utils


class ContentExtractor:
    def __init__(self):
        """Initialize the content extractor."""
        self.selectors = HTML_CONTENT_SELECTORS

    def extract_from_html(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Extract relevant content from HTML.

        Args:
            html_content: HTML content to extract from
            url: URL of the page (for reference)

        Returns:
            Dictionary with extracted content
        """
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract the page title
        title = soup.title.string if soup.title else "No Title"

        # Extract the main content
        content = self._extract_main_content(soup, url)

        # Clean the content
        cleaned_content = utils.clean_text(content)

        return {
            "url": url,
            "title": title,
            "content": cleaned_content,
            "content_snippet": cleaned_content[:300] + "..." if len(cleaned_content) > 300 else cleaned_content
        }

    def _extract_main_content(self, soup: BeautifulSoup, url: str) -> str:
        """
        Extract the main content from the parsed HTML.

        Args:
            soup: BeautifulSoup object
            url: URL of the page

        Returns:
            Extracted text content
        """
        domain = urlparse(url).netloc
        content = ""

        # Try to find the main content using different selectors
        main_content = None

        # Try each selector
        for selector in self.selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # Extract text from the main content area if found
        if main_content:
            # First try to get paragraphs
            paragraphs = main_content.find_all('p')
            if paragraphs:
                content = " ".join(p.get_text() for p in paragraphs)
            else:
                # If no paragraphs, get all text
                content = main_content.get_text()
        else:
            # Fallback: get all paragraphs from the page
            paragraphs = soup.find_all('p')
            content = " ".join(p.get_text() for p in paragraphs)

        return content

    def process_fetched_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of fetched results to extract content.

        Args:
            results: List of dictionaries with HTML content

        Returns:
            List of dictionaries with extracted content
        """
        processed_results = []

        for result in results:
            try:
                # Extract from HTML
                extracted = self.extract_from_html(result["content"], result["url"])

                # Add timestamp from original result
                extracted["timestamp"] = result.get("timestamp", utils.get_timestamp())

                processed_results.append(extracted)
            except Exception as e:
                print(f"Error processing content from {result['url']}: {e}")

        return processed_results


# Simple test function
def test_extractor():
    from web_scraper import WebScraper

    scraper = WebScraper()
    results = scraper.search_and_fetch("tomato", max_results=1)

    if results:
        extractor = ContentExtractor()
        processed = extractor.process_fetched_results(results)

        for p in processed:
            print(f"URL: {p['url']}")
            print(f"Title: {p['title']}")
            print(f"Content snippet: {p['content_snippet']}")
            print("-" * 50)


if __name__ == "__main__":
    test_extractor()