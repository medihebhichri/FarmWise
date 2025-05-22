"""
Disease parser for extracting structured information about plant diseases.
Extracts symptoms, causes, treatments, and prevention methods from content.
"""

import re
from typing import Dict, Any, List
import os

from config import (DISEASE_PATTERNS, SYMPTOM_PATTERNS, CAUSE_PATTERNS,
                    TREATMENT_PATTERNS, PREVENTION_PATTERNS)
import utils


class DiseaseParser:
    def __init__(self):
        """Initialize the disease parser."""
        self.disease_patterns = DISEASE_PATTERNS
        self.symptom_patterns = SYMPTOM_PATTERNS
        self.cause_patterns = CAUSE_PATTERNS
        self.treatment_patterns = TREATMENT_PATTERNS
        self.prevention_patterns = PREVENTION_PATTERNS

    def extract_disease_info(self, content: str, plant_name: str) -> Dict[str, Any]:
        """
        Extract structured information about plant diseases from content.

        Args:
            content: Text content to extract from
            plant_name: Name of the plant

        Returns:
            Dictionary with structured disease information
        """
        info = {
            "disease_names": [],
            "symptoms": [],
            "causes": [],
            "treatments": [],
            "prevention": []
        }

        # Extract disease names
        all_diseases = self._extract_disease_names(content, plant_name)
        if all_diseases:
            info["disease_names"] = list(set(all_diseases))

        # Extract symptoms
        info["symptoms"] = self._extract_section(content, self.symptom_patterns)

        # Extract causes
        info["causes"] = self._extract_section(content, self.cause_patterns)

        # Extract treatments
        info["treatments"] = self._extract_section(content, self.treatment_patterns)

        # Extract prevention methods
        info["prevention"] = self._extract_section(content, self.prevention_patterns)

        return info

    def _extract_disease_names(self, content: str, plant_name: str) -> List[str]:
        """
        Extract disease names from content.

        Args:
            content: Text content to extract from
            plant_name: Name of the plant

        Returns:
            List of disease names
        """
        all_diseases = []

        # Process each pattern
        for pattern in self.disease_patterns:
            # Replace {plant_name} in the pattern with the actual plant name
            if "{plant_name}" in pattern:
                pattern = pattern.replace("{plant_name}", re.escape(plant_name.capitalize()))

            # Find matches
            matches = re.findall(pattern, content)
            if isinstance(matches, list) and matches:
                # Handle cases where the result might be a tuple of groups
                if isinstance(matches[0], tuple):
                    # Extract the full match if the result is a tuple of groups
                    matches = [match[0] for match in matches]
                all_diseases.extend(matches)

        return all_diseases

    def _extract_section(self, content: str, patterns: List[str]) -> List[str]:
        """
        Extract a section (symptoms, causes, etc.) from content.

        Args:
            content: Text content to extract from
            patterns: List of regex patterns to use

        Returns:
            List of extracted items
        """
        items = []

        # Try each pattern
        for pattern in patterns:
            sections = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

            if sections:
                # Split into sentences and add to items
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', sections[0])
                items.extend([s.strip() for s in sentences if s.strip()])

        return items

    def parse_content(self, processed_results: List[Dict[str, Any]], plant_name: str) -> List[Dict[str, Any]]:
        """
        Parse a list of processed results to extract disease information.

        Args:
            processed_results: List of dictionaries with extracted content
            plant_name: Name of the plant

        Returns:
            List of dictionaries with disease information
        """
        parsed_results = []

        for result in processed_results:
            try:
                # Extract disease information
                disease_info = self.extract_disease_info(result["content"], plant_name)

                # Add disease info to the result
                result["disease_info"] = disease_info
                parsed_results.append(result)
            except Exception as e:
                print(f"Error parsing content from {result['url']}: {e}")

        return parsed_results


# Simple test function
def test_parser():
    from web_scraper import WebScraper
    from content_extractor import ContentExtractor

    scraper = WebScraper()
    results = scraper.search_and_fetch("tomato", max_results=1)

    if results:
        extractor = ContentExtractor()
        processed = extractor.process_fetched_results(results)

        parser = DiseaseParser()
        parsed = parser.parse_content(processed, "tomato")

        for p in parsed:
            print(f"URL: {p['url']}")
            print(f"Title: {p['title']}")
            print("Disease names:", p['disease_info']['disease_names'])
            print("Symptoms:", p['disease_info']['symptoms'][:2])
            print("Treatments:", p['disease_info']['treatments'][:2])
            print("-" * 50)


if __name__ == "__main__":
    test_parser()