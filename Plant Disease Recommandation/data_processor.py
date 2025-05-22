"""
Data processor for synthesizing results from multiple sources.
Combines and deduplicates information from different sources.
"""

from typing import Dict, Any, List, Optional
import os
import json

import utils


class DataProcessor:
    def __init__(self, base_dir: str = "./plant_diseases"):
        """
        Initialize the data processor.

        Args:
            base_dir: Base directory for storing results
        """
        self.base_dir = base_dir
        utils.create_directory_if_not_exists(base_dir)

    def synthesize_results(self, results: List[Dict[str, Any]], plant_name: str,
                           disease_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize results from multiple sources into a cohesive summary.

        Args:
            results: List of dictionaries with parsed disease information
            plant_name: Name of the plant
            disease_name: Specific disease name (optional)

        Returns:
            Dictionary with synthesized information
        """
        if not results:
            return {
                "status": "error",
                "message": f"No information found for {plant_name} diseases."
            }

        combined_info = {
            "plant": plant_name,
            "diseases": [],
            "sources": [{"url": r["url"], "title": r["title"]} for r in results]
        }

        if disease_name:
            # Focus on the specific disease
            disease_data = self._synthesize_specific_disease(results, disease_name)
            combined_info["diseases"].append(disease_data)
        else:
            # Combine all diseases
            all_diseases = self._synthesize_all_diseases(results)
            combined_info["diseases"] = list(all_diseases.values())

        # Save the combined information
        self._save_combined_info(combined_info, plant_name, disease_name)

        return combined_info

    def _synthesize_specific_disease(self, results: List[Dict[str, Any]],
                                     disease_name: str) -> Dict[str, Any]:
        """
        Synthesize information for a specific disease.

        Args:
            results: List of dictionaries with parsed disease information
            disease_name: Name of the disease

        Returns:
            Dictionary with synthesized disease information
        """
        disease_data = {
            "name": disease_name,
            "symptoms": [],
            "causes": [],
            "treatments": [],
            "prevention": []
        }

        for result in results:
            info = result["disease_info"]
            disease_data["symptoms"].extend(info["symptoms"])
            disease_data["causes"].extend(info["causes"])
            disease_data["treatments"].extend(info["treatments"])
            disease_data["prevention"].extend(info["prevention"])

        # Remove duplicates
        for key in ["symptoms", "causes", "treatments", "prevention"]:
            disease_data[key] = list(set(disease_data[key]))

        return disease_data

    def _synthesize_all_diseases(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Synthesize information for all diseases.

        Args:
            results: List of dictionaries with parsed disease information

        Returns:
            Dictionary mapping disease names to disease information
        """
        all_diseases = {}

        for result in results:
            disease_names = result["disease_info"]["disease_names"]

            if not disease_names:
                disease_names = ["General Disease Information"]

            for name in disease_names:
                if name not in all_diseases:
                    all_diseases[name] = {
                        "name": name,
                        "symptoms": [],
                        "causes": [],
                        "treatments": [],
                        "prevention": []
                    }

                info = result["disease_info"]
                all_diseases[name]["symptoms"].extend(info["symptoms"])
                all_diseases[name]["causes"].extend(info["causes"])
                all_diseases[name]["treatments"].extend(info["treatments"])
                all_diseases[name]["prevention"].extend(info["prevention"])

        # Remove duplicates
        for name, data in all_diseases.items():
            for key in ["symptoms", "causes", "treatments", "prevention"]:
                data[key] = list(set(data[key]))

        return all_diseases

    def _save_combined_info(self, combined_info: Dict[str, Any], plant_name: str,
                            disease_name: Optional[str] = None) -> str:
        """
        Save combined information to a JSON file.

        Args:
            combined_info: Dictionary with combined information
            plant_name: Name of the plant
            disease_name: Specific disease name (optional)

        Returns:
            Path to the saved file
        """
        plant_dir = utils.get_plant_directory(self.base_dir, plant_name)
        utils.create_directory_if_not_exists(plant_dir)

        if disease_name:
            json_filename = os.path.join(plant_dir, f"{utils.make_filename_safe(disease_name)}_info.json")
        else:
            json_filename = os.path.join(plant_dir, "all_diseases_info.json")

        utils.save_json(json_filename, combined_info)
        print(f"Combined information saved to {json_filename}")

        return json_filename

    def load_disease_info(self, plant_name: str, disease_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load disease information from a JSON file.

        Args:
            plant_name: Name of the plant
            disease_name: Specific disease name (optional)

        Returns:
            Dictionary with disease information
        """
        plant_dir = utils.get_plant_directory(self.base_dir, plant_name)

        # Check if plant directory exists
        if not os.path.exists(plant_dir):
            raise FileNotFoundError(f"No information found for plant: {plant_name}. "
                                    f"Run the search process first to gather information.")

        # Determine which JSON file to load
        if disease_name:
            json_path = os.path.join(plant_dir, f"{utils.make_filename_safe(disease_name)}_info.json")
            if not os.path.exists(json_path):
                # Try to find a match in the all_diseases file
                all_diseases_path = os.path.join(plant_dir, "all_diseases_info.json")
                if os.path.exists(all_diseases_path):
                    return self._extract_disease_from_all(all_diseases_path, disease_name)
                else:
                    raise FileNotFoundError(f"No information found for disease: {disease_name}")
        else:
            # Load general disease info
            json_path = os.path.join(plant_dir, "all_diseases_info.json")
            if not os.path.exists(json_path):
                # Look for any JSON files in the directory
                json_files = [f for f in os.listdir(plant_dir) if f.endswith('_info.json')]
                if json_files:
                    json_path = os.path.join(plant_dir, json_files[0])
                else:
                    raise FileNotFoundError(f"No disease information found for plant: {plant_name}")

        # Load and return the JSON data
        return utils.load_json(json_path)

    def _extract_disease_from_all(self, json_path: str, disease_name: str) -> Dict[str, Any]:
        """
        Extract information for a specific disease from the all_diseases JSON file.

        Args:
            json_path: Path to the all_diseases JSON file
            disease_name: Name of the disease to extract

        Returns:
            Dictionary with information for the specified disease
        """
        all_info = utils.load_json(json_path)

        # Create a new dictionary with just the specific disease
        result = {
            "plant": all_info["plant"],
            "diseases": [],
            "sources": all_info["sources"]
        }

        # Find the disease by name (case-insensitive partial match)
        disease_name_lower = disease_name.lower()
        for disease in all_info["diseases"]:
            if disease_name_lower in disease["name"].lower():
                result["diseases"].append(disease)
                return result

        # If no match found, return the original data
        print(f"Warning: Could not find exact match for disease '{disease_name}'. Returning all diseases.")
        return all_info


# Simple test function
def test_processor():
    from web_scraper import WebScraper
    from content_extractor import ContentExtractor
    from disease_parser import DiseaseParser

    scraper = WebScraper()
    results = scraper.search_and_fetch("tomato", max_results=2)

    if results:
        extractor = ContentExtractor()
        processed = extractor.process_fetched_results(results)

        parser = DiseaseParser()
        parsed = parser.parse_content(processed, "tomato")

        processor = DataProcessor()
        combined = processor.synthesize_results(parsed, "tomato")

        print(f"Combined {len(combined['diseases'])} diseases for tomato")
        for disease in combined["diseases"]:
            print(f"- {disease['name']}")


if __name__ == "__main__":
    test_processor()