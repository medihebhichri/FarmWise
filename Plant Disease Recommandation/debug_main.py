"""
Debug version of main.py with additional error handling and verbose output.
"""

import argparse
import os
import sys
import traceback
from typing import Dict, Any, Optional

# Try importing each component with error handling
try:
    print("Importing required modules...")
    from web_scraper import WebScraper
    from content_extractor import ContentExtractor
    from disease_parser import DiseaseParser
    from data_processor import DataProcessor
    from context_analyzer import ContextAnalyzer
    from recommendation_generator import RecommendationGenerator
    from report_formatter import ReportFormatter
    from user_input import UserInputCollector
    import utils

    print("All modules imported successfully!")
except ImportError as e:
    print(f"ERROR: Failed to import required module: {e}")
    print("Please ensure all Python files are in the same directory and dependencies are installed.")
    sys.exit(1)


def install_requirements():
    """Check and install required packages with verbose output."""
    print("Checking and installing required packages...")

    try:
        import requests
        print("✓ requests package is installed")
    except ImportError:
        print("Installing requests package...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
            print("✓ requests package installed successfully")
        except Exception as e:
            print(f"ERROR: Failed to install requests: {e}")

    try:
        from bs4 import BeautifulSoup
        print("✓ beautifulsoup4 package is installed")
    except ImportError:
        print("Installing beautifulsoup4 package...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])
            print("✓ beautifulsoup4 package installed successfully")
        except Exception as e:
            print(f"ERROR: Failed to install beautifulsoup4: {e}")

    try:
        import numpy as np
        print("✓ numpy package is installed")
    except ImportError:
        print("Installing numpy package...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
            print("✓ numpy package installed successfully")
        except Exception as e:
            print(f"ERROR: Failed to install numpy: {e}")

    try:
        import sklearn
        print("✓ scikit-learn package is installed")
    except ImportError:
        print("Installing scikit-learn for advanced text analysis...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
            print("✓ scikit-learn package installed successfully")
        except Exception as e:
            print(f"WARNING: Failed to install scikit-learn: {e}")
            print("Will use basic text matching instead.")


def parse_arguments():
    """Parse command line arguments with better error messages."""
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
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Check if we're in interactive mode or have required arguments
    if not args.interactive and not args.plant:
        parser.print_help()
        print("\nERROR: either --plant or --interactive is required")
        return None

    return args


def main():
    """Main entry point with enhanced error handling."""
    print("Starting Plant Disease Treatment System in debug mode...")

    # Install requirements
    install_requirements()

    # Parse command line arguments
    args = parse_arguments()
    if not args:
        return

    try:
        print(f"Creating plant disease system with base directory: {args.data_dir}")
        print(f"LM Studio URL: {args.lm_studio_url}")

        # Create the plant disease system
        system = PlantDiseaseSystem(
            base_dir=args.data_dir,
            lm_studio_url=args.lm_studio_url
        )

        # Run the system with verbose output
        print("=" * 80)
        if args.interactive:
            print("Running in interactive mode")
        else:
            print(f"Processing plant: {args.plant}")
            if args.disease:
                print(f"Disease: {args.disease}")
            if args.input_file:
                print(f"Input file: {args.input_file}")
            if args.output_file:
                print(f"Output file: {args.output_file}")
        print("=" * 80)

        system.run(
            plant_name=args.plant,
            disease_name=args.disease,
            input_file=args.input_file,
            output_file=args.output_file,
            interactive=args.interactive
        )

        print("Process completed successfully!")

    except Exception as e:
        print("\n" + "!" * 80)
        print("ERROR: An unexpected error occurred:")
        print(f"{type(e).__name__}: {e}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        print("!" * 80)
        print("\nPlease check that all files are in the correct locations and all dependencies are installed.")
        return 1

    return 0


class PlantDiseaseSystem:
    """Main system class with improved error handling."""

    def __init__(self, base_dir: str = "./plant_diseases", lm_studio_url: str = None):
        """Initialize the Plant Disease System."""
        self.base_dir = base_dir
        self.lm_studio_url = lm_studio_url

        # Create component instances
        print("Initializing system components...")
        self.scraper = WebScraper(base_dir=base_dir)
        print("✓ Web scraper initialized")

        self.extractor = ContentExtractor()
        print("✓ Content extractor initialized")

        self.parser = DiseaseParser()
        print("✓ Disease parser initialized")

        self.processor = DataProcessor(base_dir=base_dir)
        print("✓ Data processor initialized")

        self.analyzer = ContextAnalyzer()
        print("✓ Context analyzer initialized")

        self.generator = RecommendationGenerator(
            lm_studio_url=lm_studio_url) if lm_studio_url else RecommendationGenerator()
        print("✓ Recommendation generator initialized")

        self.formatter = ReportFormatter(base_dir=base_dir)
        print("✓ Report formatter initialized")

        self.input_collector = UserInputCollector()
        print("✓ User input collector initialized")

        # Ensure the base directory exists
        utils.create_directory_if_not_exists(base_dir)
        print(f"✓ Ensured base directory exists: {base_dir}")

    def collect_disease_data(self, plant_name: str, disease_name: Optional[str] = None) -> Dict[str, Any]:
        """Collect disease data from the web with better error handling."""
        print(f"\nCollecting disease information for {plant_name}...")

        try:
            # Search and fetch content from the web
            results = self.scraper.search_and_fetch(plant_name, disease_name)
            if not results:
                print("No results found.")
                return None

            print(f"Found {len(results)} results from trusted sources")

            # Extract content from HTML
            processed_results = self.extractor.process_fetched_results(results)
            print(f"Processed {len(processed_results)} results")

            # Parse content to extract disease information
            parsed_results = self.parser.parse_content(processed_results, plant_name)
            print(f"Parsed {len(parsed_results)} results")

            # Synthesize results from multiple sources
            disease_info = self.processor.synthesize_results(parsed_results, plant_name, disease_name)
            print("Successfully synthesized disease information")

            # Generate a disease report
            report_path = self.formatter.save_disease_report(disease_info)
            print(f"Disease report saved to: {report_path}")

            return disease_info

        except Exception as e:
            print(f"ERROR in collect_disease_data: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

    def load_disease_data(self, plant_name: str, disease_name: Optional[str] = None) -> Dict[str, Any]:
        """Load existing disease data with better error messages."""
        print(f"Attempting to load existing data for {plant_name}...")
        try:
            info = self.processor.load_disease_info(plant_name, disease_name)
            print("Successfully loaded existing disease information")
            return info
        except FileNotFoundError as e:
            print(f"No existing data found: {e}")
            return None
        except Exception as e:
            print(f"ERROR in load_disease_data: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

    def generate_recommendation(self, disease_info: Dict[str, Any], user_details: Dict[str, str],
                                disease_name: Optional[str] = None) -> str:
        """Generate a recommendation with better error handling."""
        print("\nAnalyzing context for personalized recommendation...")

        try:
            # Extract relevant context based on user details
            relevant_info = self.analyzer.extract_relevant_context(disease_info, user_details)
            print("Context analysis complete")

            # Generate a personalized recommendation
            print("Generating personalized recommendation...")
            recommendation = self.generator.generate_recommendation(relevant_info, user_details, disease_name)
            print("Recommendation generated")

            # Save the recommendation to a file
            report_path = self.formatter.save_recommendation_report(recommendation, disease_info["plant"], disease_name)
            print(f"Recommendation saved to: {report_path}")

            return recommendation

        except Exception as e:
            print(f"ERROR in generate_recommendation: {type(e).__name__}: {e}")
            traceback.print_exc()
            return f"Error generating recommendation: {str(e)}"

    def process_from_file(self, input_file: str, plant_name: str,
                          disease_name: Optional[str] = None) -> Dict[str, Any]:
        """Process disease information from an input file."""
        # TODO: Implement file processing
        print(f"Processing information from file: {input_file}")
        print("File processing not yet implemented. Using web search instead.")

        return self.collect_disease_data(plant_name, disease_name)

    def run_interactive(self):
        """Run the system in interactive mode with better error handling."""
        try:
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

        except Exception as e:
            print(f"ERROR in run_interactive: {type(e).__name__}: {e}")
            traceback.print_exc()

    def run(self, plant_name: str, disease_name: Optional[str] = None,
            input_file: Optional[str] = None, output_file: Optional[str] = None,
            interactive: bool = False):
        """Run the complete workflow with better error handling."""
        try:
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

        except Exception as e:
            print(f"ERROR in run: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())