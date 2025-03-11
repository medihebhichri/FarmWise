"""
recommendation_engine.py
Generates personalized plant growth recommendations by integrating data from
all components of the system.
"""

import os
import logging
import json
import requests
from typing import Dict, Any, List, Optional, Tuple

# Import configuration
from config import (
    LOGS_DIR, LLM_SERVER_URL, COMPLETIONS_ENDPOINT, DEFAULT_MODEL,
    GROWTH_FACTOR_KEYS, USER_ENV_FACTORS
)

# Import other components
from vector_database import VectorDatabase
from language_manager import LanguageManager
from environmental_analyzer import EnvironmentalAnalyzer
from data_collector import DataCollector

# Set up logging
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "recommendation_engine.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RecommendationEngine")


class RecommendationEngine:
    """
    Generates personalized plant growth recommendations by integrating
    data from multiple sources and analyzing environmental conditions.
    """

    def __init__(self):
        """Initialize the recommendation engine and its components."""
        self.vector_db = VectorDatabase()
        self.language_manager = LanguageManager()
        self.env_analyzer = EnvironmentalAnalyzer()
        self.data_collector = DataCollector()

        logger.info("Recommendation Engine initialized")

    def _ensure_plant_data_exists(self, plant_name: str, language_code: str, dialect: str = None) -> bool:
        """
        Ensure that data for the plant exists in the specified language.
        If not, collect and index it.

        Args:
            plant_name (str): Name of the plant
            language_code (str): Language code
            dialect (str, optional): Dialect code

        Returns:
            bool: True if data exists or was successfully collected
        """
        # Check if we have data for this plant
        plant_data = self.vector_db.search_by_plant(plant_name, language_code)

        if plant_data:
            logger.info(f"Found existing data for {plant_name} in {language_code}")
            return True

        logger.info(f"No data found for {plant_name} in {language_code}, collecting new data")

        # Collect data for the plant
        try:
            collected_data = self.data_collector.collect_plant_data(
                plant_name=plant_name,
                language_code=language_code,
                dialect=dialect,
                max_videos=3,
                max_websites=3
            )

            # Index the collected data
            if collected_data:
                indexed = self.vector_db.index_plant_data(collected_data)

                if indexed:
                    logger.info(f"Successfully collected and indexed data for {plant_name}")
                    return True
                else:
                    logger.error(f"Failed to index data for {plant_name}")
            else:
                logger.error(f"Failed to collect data for {plant_name}")

        except Exception as e:
            logger.error(f"Error ensuring plant data exists: {e}")

        return False

    def _get_plant_growth_factors(self, plant_name: str, language_code: str) -> Dict[str, str]:
        """
        Get growth factors for a plant from the vector database.

        Args:
            plant_name (str): Name of the plant
            language_code (str): Language code

        Returns:
            dict: Growth factors
        """
        # Try to get from vector database
        factors = self.vector_db.get_growth_factors(plant_name, language_code)

        if not factors or all(value == "Not specified" for value in factors.values()):
            logger.warning(f"No growth factors found for {plant_name} in vector database")

            # Try to find similar plants
            similar_plants = self.vector_db.find_similar_plants(plant_name, language_code)

            if similar_plants:
                similar_plant = similar_plants[0]["plant_name"]
                logger.info(f"Using growth factors from similar plant: {similar_plant}")

                factors = self.vector_db.get_growth_factors(similar_plant, language_code)

        return factors

    def _generate_detailed_care_guide(self,
                                      plant_name: str,
                                      growth_factors: Dict[str, str],
                                      language_code: str,
                                      dialect: str = None) -> str:
        """
        Generate a detailed care guide for the plant.

        Args:
            plant_name (str): Name of the plant
            growth_factors (dict): Growth factors
            language_code (str): Language code
            dialect (str, optional): Dialect code

        Returns:
            str: Detailed care guide
        """
        # Get relevant information about the plant
        plant_info = self.vector_db.search_by_plant(plant_name, language_code, top_k=5)

        # Extract text from search results
        context_text = ""
        for item in plant_info:
            context_text += item["text"] + "\n\n"

        # Create a prompt for the LLM
        lang_name = self.language_manager.get_language_name(language_code)

        prompt_templates = {
            "en": f"Create a detailed care guide for growing {plant_name}. Include sections on ideal growing conditions, planting, watering, fertilizing, common problems, and harvesting or maintenance tips. Base your guide on this information:\n\n{context_text}\n\nGrowth factors:\n",

            "ar": f"قم بإنشاء دليل عناية مفصل لزراعة {plant_name}. قم بتضمين أقسام حول ظروف النمو المثالية، والزراعة، والري، والتسميد، والمشاكل الشائعة، ونصائح الحصاد أو الصيانة. استند في دليلك على هذه المعلومات:\n\n{context_text}\n\nعوامل النمو:\n",

            "fr": f"Créez un guide d'entretien détaillé pour cultiver {plant_name}. Incluez des sections sur les conditions de culture idéales, la plantation, l'arrosage, la fertilisation, les problèmes courants et les conseils de récolte ou d'entretien. Basez votre guide sur ces informations:\n\n{context_text}\n\nFacteurs de croissance:\n"
        }

        # Default to English if language not supported
        prompt = prompt_templates.get(language_code, prompt_templates["en"])

        # Add growth factors to prompt
        for factor, value in growth_factors.items():
            if value and value != "Not specified":
                factor_title = self.language_manager.get_growth_factor_title(factor, language_code)
                prompt += f"- {factor_title}: {value}\n"

        # Complete the prompt
        prompt += f"\nWrite a comprehensive guide in {lang_name}:"

        # Generate care guide using LLM
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.4
        }

        try:
            response = requests.post(COMPLETIONS_ENDPOINT, json=payload)

            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    care_guide = data['choices'][0].get("text", "").strip()

                    # Adapt to dialect if needed
                    if dialect:
                        care_guide = self.language_manager.translate_to_dialect(care_guide, language_code, dialect)

                    return care_guide
                else:
                    logger.error("No care guide in LLM response")
            else:
                logger.error(f"LLM service error: {response.status_code}")
        except Exception as e:
            logger.error(f"Error generating care guide: {e}")

        # Fallback if generation fails
        return self.language_manager.get_text("care_guide_error", language_code, dialect)

    def generate_recommendation(self,
                                plant_name: str,
                                user_environment: Dict[str, str],
                                language_code: str = "en",
                                dialect: str = None) -> Dict[str, Any]:
        """
        Generate a personalized plant care recommendation.

        Args:
            plant_name (str): Name of the plant
            user_environment (dict): User's environment factors
            language_code (str): Language code
            dialect (str, optional): Dialect code

        Returns:
            dict: Recommendation data
        """
        logger.info(f"Generating recommendation for {plant_name} in {language_code}" +
                    (f" ({dialect} dialect)" if dialect else ""))

        # Ensure we have data for this plant
        have_data = self._ensure_plant_data_exists(plant_name, language_code, dialect)

        if not have_data:
            # Try with English as fallback
            if language_code != "en":
                logger.info(f"Trying English data for {plant_name}")
                have_data = self._ensure_plant_data_exists(plant_name, "en")

        if not have_data:
            error_message = self.language_manager.get_text("not_found", language_code, dialect, plant=plant_name)
            return {
                "status": "error",
                "message": error_message,
                "plant_name": plant_name,
                "language": language_code,
                "dialect": dialect
            }

        # Get growth factors for the plant
        growth_factors = self._get_plant_growth_factors(plant_name, language_code)

        # If factors not found in requested language, try English
        if all(value == "Not specified" for value in growth_factors.values()) and language_code != "en":
            logger.info(f"No growth factors found in {language_code}, trying English")
            growth_factors = self._get_plant_growth_factors(plant_name, "en")

        # Compare user environment with optimal growing conditions
        env_assessment = self.env_analyzer.is_environment_suitable(plant_name, user_environment, growth_factors)

        # Generate detailed care guide
        care_guide = self._generate_detailed_care_guide(plant_name, growth_factors, language_code, dialect)

        # Create recommendation text
        recommendation_text = self.language_manager.format_recommendation(
            plant_name=plant_name,
            growth_factors=growth_factors,
            environment=user_environment,
            summary=env_assessment["adaptation_tips"],
            language_code=language_code,
            dialect=dialect
        )

        # Format environment summary
        environment_summary = self.env_analyzer.get_environment_summary(user_environment, language_code)

        # Find similar plants that might be better suited to the environment
        similar_plants = self.vector_db.find_similar_plants(plant_name, language_code)

        # Compile recommendation data
        recommendation = {
            "status": "success",
            "plant_name": plant_name,
            "language": language_code,
            "dialect": dialect,
            "recommendation_text": recommendation_text,
            "environment_summary": environment_summary,
            "care_guide": care_guide,
            "growth_factors": growth_factors,
            "environment_assessment": env_assessment,
            "similar_plants": similar_plants[:3] if similar_plants else []
        }

        logger.info(f"Successfully generated recommendation for {plant_name}")
        return recommendation

    def process_plant_query(self, query: str, language_code: str = "en", dialect: str = None) -> Dict[str, Any]:
        """
        Process a natural language query about plant care.

        Args:
            query (str): User's query
            language_code (str): Language code
            dialect (str, optional): Dialect code

        Returns:
            dict: Response with relevant information
        """
        # Use LLM to extract plant name and question type
        prompt = f"Extract the plant name and question intent from this query in {language_code}. Return JSON with keys: plant_name, intent (one of: growing_info, problems, comparison, general). Query: {query}"

        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.2
        }

        try:
            response = requests.post(COMPLETIONS_ENDPOINT, json=payload)

            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    result_text = data['choices'][0].get("text", "").strip()

                    try:
                        # Parse extraction result
                        extracted = json.loads(result_text)
                        plant_name = extracted.get("plant_name", "").strip()
                        intent = extracted.get("intent", "growing_info")

                        if not plant_name:
                            # No plant name found, return general error
                            return {
                                "status": "error",
                                "message": self.language_manager.get_text("no_plant_specified", language_code, dialect),
                                "query": query
                            }

                        # Search for information based on intent
                        if intent == "growing_info":
                            # For growing info, search in vector DB
                            plant_info = self.vector_db.search_by_plant(plant_name, language_code, top_k=3)

                            if not plant_info:
                                # No info found, try to collect it
                                have_data = self._ensure_plant_data_exists(plant_name, language_code, dialect)

                                if have_data:
                                    plant_info = self.vector_db.search_by_plant(plant_name, language_code, top_k=3)
                                else:
                                    return {
                                        "status": "error",
                                        "message": self.language_manager.get_text("not_found", language_code, dialect,
                                                                                  plant=plant_name),
                                        "plant_name": plant_name,
                                        "query": query
                                    }

                            # Extract text from search results
                            context_text = ""
                            for item in plant_info:
                                context_text += item["text"] + "\n\n"

                            # Answer the query using LLM
                            answer_prompt = f"Answer this question about {plant_name} in {language_code} using only the provided information. Question: {query}\n\nInformation:\n{context_text}"

                            answer_payload = {
                                "model": DEFAULT_MODEL,
                                "prompt": answer_prompt,
                                "max_tokens": 500,
                                "temperature": 0.4
                            }

                            answer_response = requests.post(COMPLETIONS_ENDPOINT, json=answer_payload)

                            if answer_response.status_code == 200:
                                answer_data = answer_response.json()
                                if 'choices' in answer_data and len(answer_data['choices']) > 0:
                                    answer = answer_data['choices'][0].get("text", "").strip()

                                    # Adapt to dialect if needed
                                    if dialect:
                                        answer = self.language_manager.translate_to_dialect(answer, language_code,
                                                                                            dialect)

                                    return {
                                        "status": "success",
                                        "message": answer,
                                        "plant_name": plant_name,
                                        "query": query,
                                        "intent": intent
                                    }

                        # For other intents, generate specific responses
                        # (Implementation for problems, comparison, general would go here)

                        # Fallback to general plant info
                        return {
                            "status": "error",
                            "message": self.language_manager.get_text("query_not_supported", language_code, dialect),
                            "plant_name": plant_name,
                            "query": query
                        }

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing extraction result: {e}")

            # If we get here, something went wrong
            return {
                "status": "error",
                "message": self.language_manager.get_text("error_message", language_code, dialect),
                "query": query
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "status": "error",
                "message": self.language_manager.get_text("error_message", language_code, dialect),
                "query": query
            }


# For testing
if __name__ == "__main__":
    engine = RecommendationEngine()

    # Test recommendation generation
    plant_name = input("Enter plant name: ")
    language = input("Enter language code (or press Enter for English): ").strip() or "en"
    dialect = input("Enter dialect code (or press Enter for none): ").strip() or None

    # Sample user environment
    user_env = {
        "temperature": "25°C",
        "humidity": "60%",
        "soil_humidity": "Moderate",
        "wind_exposure": "Low",
        "rainfall": "Moderate",
        "sunlight_hours": "6 hours daily",
        "location_type": "indoor",
        "season": "summer"
    }

    # Generate recommendation
    recommendation = engine.generate_recommendation(
        plant_name=plant_name,
        user_environment=user_env,
        language_code=language,
        dialect=dialect
    )

    # Print recommendation
    if recommendation["status"] == "success":
        print("\n" + "=" * 50)
        print(f"Recommendation for {plant_name}:")
        print("=" * 50)
        print(recommendation["recommendation_text"])
        print("\n" + "=" * 50)
        print("Care Guide:")
        print("=" * 50)
        print(recommendation["care_guide"])
    else:
        print(f"\nError: {recommendation['message']}")