"""
User input module for gathering information about the user's situation.
"""

from typing import Dict, Any, Optional


class UserInputCollector:
    def __init__(self):
        """Initialize the user input collector."""
        pass

    def get_user_details(self) -> Dict[str, str]:
        """
        Get details from the user to help with personalized recommendations.

        Returns:
            Dictionary containing user-provided details
        """
        details = {}

        print("\n" + "=" * 60)
        print("PLANT DISEASE TREATMENT - PERSONALIZATION")
        print("=" * 60)
        print("Please provide some details to help personalize your treatment recommendation.")
        print("Press Enter to skip any question if you're unsure.\n")

        # Get location/climate information
        details["location"] = input(
            "What is your location or growing zone? (e.g., 'Zone 7b', 'Mediterranean climate'): ")

        # Get information about the growing environment
        details["environment"] = input(
            "Where is the plant growing? (e.g., 'indoor potted', 'outdoor garden', 'greenhouse'): ")

        # Get information about organic preferences
        organic_pref = input("Do you prefer organic solutions? (yes/no): ").lower()
        details["organic_preferred"] = organic_pref.startswith('y')

        # Get severity level
        severity = input("How severe is the disease? (mild/moderate/severe): ").lower()
        details["severity"] = severity if severity in ['mild', 'moderate', 'severe'] else 'unknown'

        # Get information about other plants
        details["other_plants"] = input("Are there other plants nearby that could be affected? (yes/no + details): ")

        # Get any previous treatments applied
        details["previous_treatments"] = input("Have you applied any treatments already? If so, what?: ")

        print("\nThank you for providing these details. This will help customize your treatment recommendation.")

        return details

    def get_plant_and_disease_info(self) -> Dict[str, str]:
        """
        Get information about the plant and disease from the user.

        Returns:
            Dictionary containing plant and disease information
        """
        info = {}

        print("\n" + "=" * 60)
        print("PLANT DISEASE IDENTIFICATION")
        print("=" * 60)

        # Get plant name
        while True:
            plant_name = input("What plant is affected? (e.g., 'tomato', 'rose', 'apple'): ").strip()
            if plant_name:
                info["plant_name"] = plant_name
                break
            print("Plant name is required. Please try again.")

        # Get disease name (optional)
        disease_name = input("If you know the specific disease, enter it (or press Enter to skip): ").strip()
        if disease_name:
            info["disease_name"] = disease_name

        # Get input file path (optional)
        input_file = input("Path to input file (optional, press Enter to skip): ").strip()
        if input_file:
            info["input_file"] = input_file

        # Get output file path (optional)
        output_file = input("Path for output recommendation file (optional, press Enter for default): ").strip()
        if output_file:
            info["output_file"] = output_file

        return info


def test_input_collector():
    collector = UserInputCollector()

    # Get plant and disease info
    plant_info = collector.get_plant_and_disease_info()
    print("\nPlant Information:")
    for key, value in plant_info.items():
        print(f"  {key}: {value}")

    # Get user details
    user_details = collector.get_user_details()
    print("\nUser Details:")
    for key, value in user_details.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_input_collector()