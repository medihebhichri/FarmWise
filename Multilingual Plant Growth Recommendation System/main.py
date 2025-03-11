"""
main.py
Main entry point for the Plant Growth Recommendation System.
Coordinates all components and provides a user interface.
"""

import os
import sys
import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple

# Import configuration
from config import (
    LOGS_DIR, SUPPORTED_LANGUAGES, DATA_DIR, VECTOR_DB_DIR,
    USER_ENV_FACTORS
)

# Import components
from data_collector import DataCollector
from vector_database import VectorDatabase
from language_manager import LanguageManager
from environmental_analyzer import EnvironmentalAnalyzer
from recommendation_engine import RecommendationEngine

# Set up logging
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "main.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")


class PlantRecommendationSystem:

    def __init__(self):

        logger.info("Initializing Plant Growth Recommendation System")
        self.data_collector = DataCollector()
        self.vector_db = VectorDatabase()
        self.language_manager = LanguageManager()
        self.env_analyzer = EnvironmentalAnalyzer()
        self.recommendation_engine = RecommendationEngine()
        self.current_language = "en"
        self.current_dialect = None
        self.user_environment = {factor: "Not specified" for factor in USER_ENV_FACTORS}

        logger.info("System initialized successfully")

    def display_welcome(self):
        welcome = self.language_manager.get_text("greeting", self.current_language, self.current_dialect)
        print("\n" + "=" * 60)
        print(welcome)
        print("=" * 60 + "\n")

    def set_language(self, language_code: str, dialect: str = None):
        if language_code in SUPPORTED_LANGUAGES:
            self.current_language = language_code

            # Validate dialect
            if dialect:
                valid_dialects = self.language_manager.get_dialects_for_language(language_code)
                if dialect in valid_dialects:
                    self.current_dialect = dialect
                else:
                    logger.warning(f"Invalid dialect '{dialect}' for language '{language_code}'")
                    self.current_dialect = None
            else:
                self.current_dialect = None

            language_name = self.language_manager.get_language_name(language_code)
            logger.info(f"Language set to {language_name}" +
                        (f" ({dialect} dialect)" if self.current_dialect else ""))
        else:
            logger.warning(f"Unsupported language code: {language_code}")

    def get_supported_languages(self) -> Dict[str, Dict[str, Any]]:

        return SUPPORTED_LANGUAGES

    def collect_environment_info(self):
        """Collect information about the user's growing environment."""
        print("\n" + "=" * 60)
        env_question = self.language_manager.get_text("environment_question",
                                                      self.current_language,
                                                      self.current_dialect)
        print(env_question)
        print("=" * 60)

        print(self.language_manager.get_text("environment_prompt",
                                             self.current_language,
                                             self.current_dialect))

        user_input = input("> ")
        parsed_env = self.env_analyzer.parse_user_environment(user_input)
        self.user_environment = parsed_env
        env_summary = self.env_analyzer.get_environment_summary(
            parsed_env, self.current_language
        )

        print("\n" + env_summary + "\n")

    def get_plant_recommendation(self, plant_name: str) -> Dict[str, Any]:
        recommendation = self.recommendation_engine.generate_recommendation(
            plant_name=plant_name,
            user_environment=self.user_environment,
            language_code=self.current_language,
            dialect=self.current_dialect
        )

        return recommendation

    def display_recommendation(self, recommendation: Dict[str, Any]):

        if recommendation["status"] == "error":
            print(f"\n{recommendation['message']}")
            return
        print("\n" + "=" * 60)
        plant_name = recommendation["plant_name"]
        print(
            f"{self.language_manager.get_text('recommendation_title', self.current_language, self.current_dialect, plant=plant_name)}")
        print("=" * 60 + "\n")

        print(recommendation["recommendation_text"])

        print("\n" + "=" * 60)
        print(f"{self.language_manager.get_text('care_guide_title', self.current_language, self.current_dialect)}")
        print("=" * 60 + "\n")

        print(recommendation["care_guide"])

        if recommendation["similar_plants"]:
            print("\n" + "=" * 60)
            print(
                f"{self.language_manager.get_text('similar_plants_title', self.current_language, self.current_dialect)}")
            print("=" * 60 + "\n")

            for i, plant in enumerate(recommendation["similar_plants"]):
                print(f"{i + 1}. {plant['plant_name']}")

    def process_natural_language_query(self, query: str) -> Dict[str, Any]:
        response = self.recommendation_engine.process_plant_query(
            query=query,
            language_code=self.current_language,
            dialect=self.current_dialect
        )

        return response

    def display_query_response(self, response: Dict[str, Any]):
        if response["status"] == "error":
            print(f"\n{response['message']}")
            return

        print("\n" + "=" * 60)
        if "plant_name" in response:
            print(f"{self.language_manager.get_text('query_response_title', self.current_language, self.current_dialect, plant=response['plant_name'])}")
        else:
            print(f"{self.language_manager.get_text('query_response_title_generic', self.current_language, self.current_dialect)}")
        print("=" * 60 + "\n")

        print(response["message"])

    def run_interactive_mode(self):
        self.display_welcome()
        print("\nAvailable languages:")
        for code, info in SUPPORTED_LANGUAGES.items():
            if info.get("active", True):
                print(f"- {info['name']} ({code})")

        lang_code = input(f"\nSelect language code [{self.current_language}]: ").strip() or self.current_language
        self.set_language(lang_code)

        dialects = self.language_manager.get_dialects_for_language(self.current_language)
        if dialects:
            print(f"\nAvailable dialects for {self.language_manager.get_language_name(self.current_language)}:")
            for dialect in dialects:
                print(f"- {dialect}")

            dialect_code = input("Select dialect (or press Enter for none): ").strip() or None
            if dialect_code:
                self.set_language(self.current_language, dialect_code)
        self.collect_environment_info()
        while True:
            print("\n" + "=" * 60)
            print(self.language_manager.get_text("main_menu", self.current_language, self.current_dialect))
            print("=" * 60)
            print("1. " + self.language_manager.get_text("menu_plant_recommendation", self.current_language, self.current_dialect))
            print("2. " + self.language_manager.get_text("menu_ask_question", self.current_language, self.current_dialect))
            print("3. " + self.language_manager.get_text("menu_change_language", self.current_language, self.current_dialect))
            print("4. " + self.language_manager.get_text("menu_update_environment", self.current_language, self.current_dialect))
            print("5. " + self.language_manager.get_text("menu_exit", self.current_language, self.current_dialect))

            choice = input("\n> ").strip()

            if choice == "1":
                plant_name = input(f"\n{self.language_manager.get_text('plant_question', self.current_language, self.current_dialect)} ")

                if plant_name.strip():
                    print(f"\n{self.language_manager.get_text('generating_recommendation', self.current_language, self.current_dialect)}")
                    recommendation = self.get_plant_recommendation(plant_name)
                    self.display_recommendation(recommendation)

            elif choice == "2":
                query = input(f"\n{self.language_manager.get_text('query_question', self.current_language, self.current_dialect)} ")

                if query.strip():
                    print(f"\n{self.language_manager.get_text('processing_query', self.current_language, self.current_dialect)}")
                    response = self.process_natural_language_query(query)
                    self.display_query_response(response)

            elif choice == "3":
                print("\nAvailable languages:")
                for code, info in SUPPORTED_LANGUAGES.items():
                    if info.get("active", True):
                        print(f"- {info['name']} ({code})")

                lang_code = input(f"\nSelect language code [{self.current_language}]: ").strip() or self.current_language
                self.set_language(lang_code)
                dialects = self.language_manager.get_dialects_for_language(self.current_language)
                if dialects:
                    print(f"\nAvailable dialects for {self.language_manager.get_language_name(self.current_language)}:")
                    for dialect in dialects:
                        print(f"- {dialect}")

                    dialect_code = input("Select dialect (or press Enter for none): ").strip() or None
                    if dialect_code:
                        self.set_language(self.current_language, dialect_code)

            elif choice == "4":
                # Update environment
                self.collect_environment_info()

            elif choice == "5":
                # Exit
                print(f"\n{self.language_manager.get_text('goodbye', self.current_language, self.current_dialect)}")
                break

            else:
                print(f"\n{self.language_manager.get_text('invalid_choice', self.current_language, self.current_dialect)}")


def create_index_database():
    print("Initializing vector database...")
    vector_db = VectorDatabase()

    print("Indexing all plant data files...")
    indexed_count = vector_db.index_all_plant_files()

    print(f"Indexed {indexed_count} plant data files.")
    print(f"Vector database now contains {vector_db.index.ntotal if vector_db.index else 0} vectors")


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "index":
            create_index_database()
            return
        elif command == "collect":
            if len(sys.argv) > 2:
                plants = sys.argv[2].split(",")
                language = sys.argv[3] if len(sys.argv) > 3 else "en"
                dialect = sys.argv[4] if len(sys.argv) > 4 else None

                collector = DataCollector()
                collector.collect_data_for_multiple_plants(
                    plant_list=plants,
                    language_code=language,
                    dialect=dialect
                )
            else:
                print("Usage: python main.py collect plant1,plant2,plant3 [language] [dialect]")
            return
    system = PlantRecommendationSystem()
    system.run_interactive_mode()


if __name__ == "__main__":
    main()