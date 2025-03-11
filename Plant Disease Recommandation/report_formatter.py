"""
Report formatter for creating markdown reports from plant disease information.
Formats disease information and recommendations into readable reports.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import os

import utils


class ReportFormatter:
    def __init__(self, base_dir: str = "./plant_diseases"):
        """
        Initialize the report formatter.

        Args:
            base_dir: Base directory for storing reports
        """
        self.base_dir = base_dir
        utils.create_directory_if_not_exists(base_dir)

    def format_disease_report(self, disease_info: Dict[str, Any]) -> str:
        """
        Format disease information into a markdown report.

        Args:
            disease_info: Dictionary containing disease information

        Returns:
            Markdown formatted report
        """
        plant_name = disease_info["plant"]
        diseases = disease_info.get("diseases", [])
        sources = disease_info.get("sources", [])

        # Create the report header
        report = f"# Plant Disease Report: {plant_name.capitalize()}\n\n"
        report += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

        if not diseases:
            report += "No specific disease information found.\n"
            return report

        # Add table of contents for multiple diseases
        if len(diseases) > 1:
            report += "## Table of Contents\n\n"
            for i, disease in enumerate(diseases, 1):
                anchor = disease['name'].lower().replace(' ', '-')
                report += f"{i}. [{disease['name']}](#{anchor})\n"
            report += "\n"

        # Add each disease section
        for disease in diseases:
            anchor = disease['name'].lower().replace(' ', '-')
            report += f"## {disease['name']}\n\n"

            if disease.get("symptoms"):
                report += "### Symptoms\n\n"
                for symptom in disease["symptoms"]:
                    report += f"- {symptom}\n"
                report += "\n"

            if disease.get("causes"):
                report += "### Causes\n\n"
                for cause in disease["causes"]:
                    report += f"- {cause}\n"
                report += "\n"

            if disease.get("treatments"):
                report += "### Treatment\n\n"
                for treatment in disease["treatments"]:
                    report += f"- {treatment}\n"
                report += "\n"

            if disease.get("prevention"):
                report += "### Prevention\n\n"
                for prevention in disease["prevention"]:
                    report += f"- {prevention}\n"
                report += "\n"

        # Add sources
        if sources:
            report += "## Sources\n\n"
            for source in sources:
                title = source.get("title", source.get("url", "Unknown Source"))
                url = source.get("url", "#")
                report += f"- [{title}]({url})\n"

        return report

    def format_recommendation_report(self, recommendation: str, plant_name: str,
                                     disease_name: Optional[str] = None) -> str:
        """
        Format a recommendation into a markdown report.

        Args:
            recommendation: Recommendation text
            plant_name: Name of the plant
            disease_name: Specific disease name (optional)

        Returns:
            Markdown formatted report
        """
        # Create the report header
        report = f"# Treatment Recommendation: {plant_name.capitalize()}"
        if disease_name:
            report += f" - {disease_name}"
        report += "\n\n"

        report += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

        # Add the recommendation
        report += recommendation

        # Add footer
        report += "\n\n---\n"
        report += "*This recommendation is based on information gathered from trusted sources. "
        report += "Always consult with a local extension service or professional for specific advice.*\n"

        return report

    def save_disease_report(self, disease_info: Dict[str, Any]) -> str:
        """
        Format and save a disease report.

        Args:
            disease_info: Dictionary containing disease information

        Returns:
            Path to the saved report
        """
        plant_name = disease_info["plant"]
        plant_dir = utils.get_plant_directory(self.base_dir, plant_name)
        utils.create_directory_if_not_exists(plant_dir)

        # Generate the report content
        report_content = self.format_disease_report(disease_info)

        # Determine the filename
        diseases = disease_info.get("diseases", [])
        if len(diseases) == 1 and diseases[0]["name"] != "General Disease Information":
            filename = f"{utils.make_filename_safe(diseases[0]['name'])}_report.md"
        else:
            filename = "disease_report.md"

        report_path = os.path.join(plant_dir, filename)

        # Save the report
        with open(report_path, 'w') as f:
            f.write(report_content)

        print(f"Report saved to {report_path}")
        return report_path

    def save_recommendation_report(self, recommendation: str, plant_name: str,
                                   disease_name: Optional[str] = None) -> str:
        """
        Format and save a recommendation report.

        Args:
            recommendation: Recommendation text
            plant_name: Name of the plant
            disease_name: Specific disease name (optional)

        Returns:
            Path to the saved report
        """
        plant_dir = utils.get_plant_directory(self.base_dir, plant_name)
        utils.create_directory_if_not_exists(plant_dir)

        # Generate the report content
        report_content = self.format_recommendation_report(recommendation, plant_name, disease_name)

        # Determine the filename
        if disease_name:
            filename = f"{utils.make_filename_safe(disease_name)}_recommendation.md"
        else:
            filename = "treatment_recommendation.md"

        report_path = os.path.join(plant_dir, filename)

        # Save the report
        with open(report_path, 'w') as f:
            f.write(report_content)

        print(f"Recommendation saved to {report_path}")
        return report_path


# Simple test function
def test_formatter():
    from data_processor import DataProcessor

    processor = DataProcessor()

    try:
        # Try to load existing data
        info = processor.load_disease_info("tomato")

        formatter = ReportFormatter()
        report_path = formatter.save_disease_report(info)

        print(f"Test report saved to: {report_path}")

        # Create a mock recommendation
        recommendation = """
## DIAGNOSIS CONFIRMATION
Based on the symptoms described, this appears to be Early Blight. The symptoms match the typical presentation of this disease.

## IMMEDIATE TREATMENT
Apply copper-based organic fungicide according to package directions. Repeat every 7-10 days as needed.

## LONG-TERM MANAGEMENT
1. Practice crop rotation, waiting at least 2 years before planting tomatoes in the same location
2. Use resistant varieties in future plantings
3. Apply compost to help suppress soil-borne pathogens

## CULTURAL PRACTICES
- Water early in the day at the base of plants to keep foliage dry
- Space plants properly to improve air circulation
- Remove lower leaves that are close to the soil

## ADDITIONAL CONSIDERATIONS
- Monitor nearby plants closely as the disease can spread
- Sanitize garden tools after use
- Consider using straw mulch to prevent soil from splashing onto leaves
"""

        rec_path = formatter.save_recommendation_report(recommendation, "tomato", "Early Blight")
        print(f"Test recommendation saved to: {rec_path}")

    except FileNotFoundError:
        print("No existing data found. Run the data processor first.")


if __name__ == "__main__":
    test_formatter()