"""
Main module for the Plant Disease System.
Coordinates the entire workflow from data collection to recommendation generation.
"""

import argparse
import os
import sys
from typing import Dict, Any, Optional

# Import all components
from web_scraper import WebScraper
from content_extractor import ContentExtractor
from disease_parser import DiseaseParser
from data_processor import DataProcessor
from context_analyzer import ContextAnalyzer
from recommendation_generator import RecommendationGenerator
from report_formatter import ReportFormatter
from user_input import UserInputCollector
import utils


class PlantDiseaseSystem:
    def __init__(self, base_dir: str = "./plant_diseases", lm_studio_url: str = None):
        """
        Initialize the Plant Disease System.

        Args:
            base_dir: Base directory for storing data and reports
            lm_studio_url: URL for LM Studio API (optional)
        """
        self.base_dir = base_dir
        self.lm_studio_url = lm_studio_url

        # Create component instances
        self.scraper = WebScraper(base_dir=base_dir)
        self.extractor = ContentExtractor()
        self.parser = DiseaseParser()
        self.processor = DataProcessor(base_dir=base_dir)
        self.analyzer = ContextAnalyzer()
        self.generator = RecommendationGenerator(
            lm_studio_url=lm_studio_url) if lm_studio_url else RecommendationGenerator()
        self.formatter = ReportFormatter(base_dir=base_dir)
        self.input_collector = UserInputCollector()

        # Ensure the base directory exists
        utils.create_directory_if_not_exists(base_dir)

    def collect_disease_data(self, plant_name: str, disease_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect disease data from the web.

        Args:
            plant_name: Name of the plant
            disease_name: Specific disease name (optional)

        Returns:
            Dictionary with collected and processed disease information
        """
        print(f"\nCollecting disease information for {plant_name}...")

        # Search and fetch content from the web
        results = self.scraper.search_and_fetch(plant_name, disease_name)
        if not results:
            print("No results found.")
            return None

        # Extract content from HTML
        processed_results = self.extractor.process_fetched_results(results)

        # Parse content to extract disease information
        parsed_results = self.parser.parse_content(processed_results, plant_name)

        # Synthesize results from multiple sources
        disease_info = self.processor.synthesize_results(parsed_results, plant_name, disease_name)

        # Generate a disease report
        self.formatter.save_disease_report(disease_info)

        return disease_info

    def load_disease_data(self, plant_name: str, disease_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load existing disease data.

        Args:
            plant_name: Name of the plant
            disease_name: Specific disease name (optional)

        Returns:
            Dictionary with disease information
        """
        try:
            return self.processor.load_disease_info(plant_name, disease_name)
        except FileNotFoundError:
            print(f"No existing data found for {plant_name}.")
            return None

    def generate_recommendation(self, disease_info: Dict[str, Any], user_details: Dict[str, str],
                                disease_name: Optional[str] = None) -> str:
        """
        Generate a recommendation based on disease information and user details.

        Args:
            disease_info: Dictionary with disease information
            user_details: Dictionary with user details
            disease_name: Specific disease name (optional)

        Returns:
            Generated recommendation text
        """
        print("\nAnalyzing context for personalized recommendation...")

        # Extract relevant context based on user details
        relevant_info = self.analyzer.extract_relevant_context(disease_info, user_details)

        # Generate a personalized recommendation
        recommendation = self.generator.generate_recommendation(relevant_info, user_details, disease_name)

        # Save the recommendation to a file
        self.formatter.save_recommendation_report(recommendation, disease_info["plant"], disease_name)

        return recommendation

    def process_from_file(self, input_file: str, plant_name: str,
                          disease_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process disease information from an input file.

        Args:
            input_file: Path to the input file
            plant_name: Name of the plant
            disease_name: Specific disease name (optional)

        Returns:
            Dictionary with disease information
        """
        # TODO: Implement file processing
        print(f"Processing information from file: {input_file}")
        print("File processing not yet implemented. Using web search instead.")

        return self.collect_disease_data(plant_name, disease_name)

    def run_interactive(self):
        """Run the system in interactive mode."""
        print("\n" + "=" * 80)
        print("PLANT DISEASE TREATMENT RECOMMENDATION SYSTEM")
        print("=" * 80)
        print("This system helps diagnose plant diseases and provides personalized treatment recommendations.")

        # Get plant and disease information
        plant_info = self.input_collector.get_plant_and_disease_info()
        plant_name = plant_info["plant_name"]
        disease_name = plant_info.get("disease_name")
        input_file = plant_info.get("input_file")

        # Load or collect disease information
        disease_info = self.load_disease_data(plant_name, disease_name)

        if not disease_info:
            print("No existing data found. Collecting information from the web...")
            if input_file:
                disease_info = self.process_from_file(input_file, plant_name, disease_name)
            else:
                disease_info = self.collect_disease_data(plant_name, disease_name)

        if not disease_info:
            print("Failed to collect disease information. Exiting.")
            return

        # Get user details for personalized recommendation
        user_details = self.input_collector.get_user_details()

        # Generate personalized recommendation
        recommendation = self.generate_recommendation(disease_info, user_details, disease_name)

        # Print the recommendation
        print("\n" + "=" * 80)
        print("PERSONALIZED TREATMENT RECOMMENDATION")
        print("=" * 80 + "\n")
        print(recommendation)

        print("\n" + "=" * 80)
        print("Search complete! Results are organized in the following directory structure:")
        print(f"{self.base_dir}/")
        print(f"└── {plant_name.lower().replace(' ', '_')}/")
        print(f"    ├── cache/                        # Cached web pages")
        print(f"    ├── disease_report.md             # Disease information report")
        print(f"    ├── treatment_recommendation.md   # Personalized recommendation")
        print(f"    └── all_diseases_info.json        # Structured disease data")

    def run(self, plant_name: str, disease_name: Optional[str] = None,
            input_file: Optional[str] = None, output_file: Optional[str] = None,
            interactive: bool = False):
        """
        Run the complete workflow.

        Args:
            plant_name: Name of the plant
            disease_name: Specific disease name (optional)
            input_file: Path to the input file (optional)
            output_file: Path to the output file (optional)
            interactive: Whether to run in interactive mode
        """
        if interactive:
            self.run_interactive()
            return

        # Load or collect disease information
        disease_info = self.load_disease_data(plant_name, disease_name)

        if not disease_info:
            print("No existing data found. Collecting information from the web...")
            if input_file:
                disease_info = self.process_from_file(input_file, plant_name, disease_name)
            else:
                disease_info = self.collect_disease_data(plant_name, disease_name)

        if not disease_info:
            print("Failed to collect disease information. Exiting.")
            return

        # Get user details for personalized recommendation
        user_details = self.input_collector.get_user_details()

        # Generate personalized recommendation
        recommendation = self.generate_recommendation(disease_info, user_details, disease_name)

        # Print the recommendation
        print("\n" + "=" * 80)
        print("PERSONALIZED TREATMENT RECOMMENDATION")
        print("=" * 80 + "\n")
        print(recommendation)

        # Save to output file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(recommendation)
            print(f"Recommendation saved to {output_file}")


def parse_arguments():
    """Parse command line arguments.

    Returns:
        Parsed arguments or None if invalid
    """
    parser = argparse.ArgumentParser(
        description='Generate personalized plant disease treatment recommendations')
    parser.add_argument('--plant', help='Name of the plant')
    parser.add_argument('--disease', help='Specific disease name (optional)')
    parser.add_argument('--input-file', help='Path to input file (optional)')
    parser.add_argument('--output-file', help='Output file for the recommendation (optional)')
    parser.add_argument('--data-dir', help='Directory for plant disease data (default: ./plant_diseases)',
                        default='./plant_diseases')
    parser.add_argument('--lm-studio-url', help='URL for LM Studio API (default: http://localhost:1234/v1)',
                        default='http://localhost:1234/v1')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')

    args = parser.parse_args()

    # Check if we're in interactive mode or have required arguments
    if not args.interactive and not args.plant:
        parser.print_help()
        print("\nError: either --plant or --interactive is required")
        return None

    return args