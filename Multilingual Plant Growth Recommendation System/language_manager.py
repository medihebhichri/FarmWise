"""
language_manager.py
Handles translation and language-specific processing for the plant recommendation system.
Supports multiple languages and dialects, including Tunisian Arabic.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple

# Import configuration
from config import (
    LOGS_DIR, SUPPORTED_LANGUAGES, TRANSLATION_API_KEY,
    USE_TRANSLATION_SERVICE, DIALECT_ADJUSTMENTS,
    LLM_SERVER_URL, COMPLETIONS_ENDPOINT, DEFAULT_MODEL
)

# Set up logging
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "language_manager.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LanguageManager")

class LanguageManager:
    """Manages language processing, translation, and dialect support."""

    def __init__(self):
        """Initialize the language manager."""
        self.default_language = "en"

        # Load language templates and phrases
        self._load_language_templates()

    def _load_language_templates(self):
        """Load language-specific templates and phrases."""
        # This would typically load from files, but for now we'll define inline
        self.templates = {
            # English templates
            "en": {
                "greeting": "Welcome! I'll help you with plant care information.",
                "plant_question": "What plant would you like information about?",
                "language_question": "What language would you prefer?",
                "dialect_question": "Which dialect would you prefer?",
                "environment_question": "Please tell me about your growing environment.",
                "recommendation_intro": "Here are my recommendations for growing {plant}:",
                "not_found": "I don't have information about {plant} yet.",
                "error_message": "Sorry, I encountered an error. Please try again.",
                "growth_factor_titles": {
                    "temperature": "Temperature",
                    "humidity": "Humidity",
                    "soil_type": "Soil Type",
                    "planting_method": "Planting Method",
                    "care_routine": "Care Routine",
                    "light_exposure": "Light Exposure",
                    "water_requirements": "Watering Needs",
                    "fertilizer": "Fertilizer",
                    "pH_level": "pH Level",
                    "plant_spacing": "Plant Spacing"
                }
            },

            # Arabic templates
            "ar": {
                "greeting": "مرحبا! سأساعدك بمعلومات عن العناية بالنباتات.",
                "plant_question": "ما هو النبات الذي تريد معلومات عنه؟",
                "language_question": "ما هي اللغة التي تفضلها؟",
                "dialect_question": "ما هي اللهجة التي تفضلها؟",
                "environment_question": "من فضلك أخبرني عن بيئة النمو لديك.",
                "recommendation_intro": "إليك توصياتي لزراعة {plant}:",
                "not_found": "ليس لدي معلومات عن {plant} بعد.",
                "error_message": "آسف، لقد واجهت خطأ. يرجى المحاولة مرة أخرى.",
                "growth_factor_titles": {
                    "temperature": "درجة الحرارة",
                    "humidity": "الرطوبة",
                    "soil_type": "نوع التربة",
                    "planting_method": "طريقة الزراعة",
                    "care_routine": "روتين العناية",
                    "light_exposure": "التعرض للضوء",
                    "water_requirements": "متطلبات الري",
                    "fertilizer": "السماد",
                    "pH_level": "مستوى الحموضة",
                    "plant_spacing": "المسافة بين النباتات"
                }
            },

            # French templates
            "fr": {
                "greeting": "Bienvenue! Je vais vous aider avec des informations sur l'entretien des plantes.",
                "plant_question": "Sur quelle plante souhaitez-vous des informations?",
                "language_question": "Quelle langue préférez-vous?",
                "dialect_question": "Quel dialecte préférez-vous?",
                "environment_question": "Veuillez me parler de votre environnement de culture.",
                "recommendation_intro": "Voici mes recommandations pour cultiver {plant}:",
                "not_found": "Je n'ai pas encore d'informations sur {plant}.",
                "error_message": "Désolé, j'ai rencontré une erreur. Veuillez réessayer.",
                "growth_factor_titles": {
                    "temperature": "Température",
                    "humidity": "Humidité",
                    "soil_type": "Type de Sol",
                    "planting_method": "Méthode de Plantation",
                    "care_routine": "Routine d'Entretien",
                    "light_exposure": "Exposition à la Lumière",
                    "water_requirements": "Besoins en Eau",
                    "fertilizer": "Engrais",
                    "pH_level": "Niveau de pH",
                    "plant_spacing": "Espacement des Plantes"
                }
            }
        }

        # Dialect-specific adjustments
        self.dialect_templates = {
            # Tunisian Arabic dialect
            "ar_TN": {
                "greeting": "مرحبا بيك! باش نعاونك بمعلومات على النباتات.",
                "plant_question": "شنوة النبتة إلي تحب معلومات عليها؟",
                "environment_question": "عطيني معلومات على البيئة متاعك باش تزرع.",
                "recommendation_intro": "هاو توصياتي باش تزرع {plant}:",
                "not_found": "ماعنديش معلومات على {plant} حتى الآن.",
                "error_message": "سامحني، صارت غلطة. حاول مرة أخرى."
            },

            # Egyptian Arabic dialect
            "ar_EG": {
                "greeting": "أهلا بيك! هساعدك بمعلومات عن النباتات.",
                "plant_question": "إيه النبات اللي عايز معلومات عنه؟",
                "environment_question": "قولي عن بيئة الزراعة عندك.",
                "recommendation_intro": "دي توصياتي عشان تزرع {plant}:",
                "not_found": "معنديش معلومات عن {plant} دلوقتي.",
                "error_message": "آسف، حصل مشكلة. حاول تاني."
            },

            # Tunisian French dialect
            "fr_TN": {
                "greeting": "Ahla! Je vais t'aider avec des informations sur les plantes.",
                "plant_question": "C'est quoi la plante que tu veux connaître?",
                "environment_question": "Dis-moi comment c'est chez toi pour faire pousser des plantes.",
                "recommendation_intro": "Voilà mes conseils pour cultiver {plant}:",
                "not_found": "Je n'ai pas d'infos sur {plant} pour le moment.",
                "error_message": "Désolé, il y a eu une erreur. Essaie encore."
            }
        }

    def get_supported_languages(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of supported languages and dialects.

        Returns:
            dict: Dictionary of supported languages and dialects
        """
        return SUPPORTED_LANGUAGES

    def get_language_name(self, language_code: str) -> str:
        """
        Get the full name of a language from its code.

        Args:
            language_code (str): Language code

        Returns:
            str: Language name or 'Unknown language'
        """
        if language_code in SUPPORTED_LANGUAGES:
            return SUPPORTED_LANGUAGES[language_code]["name"]
        return "Unknown language"

    def get_dialects_for_language(self, language_code: str) -> List[str]:
        """
        Get available dialects for a language.

        Args:
            language_code (str): Language code

        Returns:
            list: List of dialect codes
        """
        if language_code in SUPPORTED_LANGUAGES:
            return SUPPORTED_LANGUAGES[language_code].get("dialects", [])
        return []

    def get_text(self, key: str, language_code: str, dialect: str = None, **kwargs) -> str:
        """
        Get text in the specified language and dialect.

        Args:
            key (str): Text key
            language_code (str): Language code
            dialect (str, optional): Dialect code
            **kwargs: Format parameters

        Returns:
            str: Localized text
        """
        # Check for dialect-specific template
        dialect_key = f"{language_code}_{dialect}" if dialect else None

        if dialect_key and dialect_key in self.dialect_templates and key in self.dialect_templates[dialect_key]:
            text = self.dialect_templates[dialect_key][key]
        elif language_code in self.templates and key in self.templates[language_code]:
            text = self.templates[language_code][key]
        else:
            # Fall back to English
            text = self.templates.get("en", {}).get(key, f"[{key}]")

        # Format text with provided parameters
        if kwargs:
            try:
                text = text.format(**kwargs)
            except Exception as e:
                logger.error(f"Error formatting text '{key}': {e}")

        return text

    def get_growth_factor_title(self, factor: str, language_code: str) -> str:
        """
        Get localized title for a growth factor.

        Args:
            factor (str): Growth factor key
            language_code (str): Language code

        Returns:
            str: Localized title
        """
        titles = self.templates.get(language_code, {}).get("growth_factor_titles", {})
        return titles.get(factor, self.templates["en"]["growth_factor_titles"].get(factor, factor))

    def translate_text(self, text: str, source_language: str, target_language: str) -> str:
        """
        Translate text between languages.

        Args:
            text (str): Text to translate
            source_language (str): Source language code
            target_language (str): Target language code

        Returns:
            str: Translated text
        """
        # Skip translation if languages are the same
        if source_language == target_language:
            return text

        # Use external translation service if enabled
        if USE_TRANSLATION_SERVICE and TRANSLATION_API_KEY:
            try:
                # This would integrate with a service like Google Translate, Azure Translator, etc.
                # Placeholder implementation
                logger.info(f"Using external service to translate from {source_language} to {target_language}")
                translated = self._translate_with_external_service(text, source_language, target_language)
                return translated
            except Exception as e:
                logger.error(f"Error with external translation service: {e}")
                # Fall back to LLM translation

        # Use LLM for translation
        return self._translate_with_llm(text, source_language, target_language)

    def _translate_with_external_service(self, text: str, source_language: str, target_language: str) -> str:
        """
        Translate text using an external translation service.

        Args:
            text (str): Text to translate
            source_language (str): Source language code
            target_language (str): Target language code

        Returns:
            str: Translated text
        """
        # This would implement a specific translation API
        # Placeholder implementation
        api_url = "https://translation-service.example.com/translate"
        headers = {"Authorization": f"Bearer {TRANSLATION_API_KEY}"}
        payload = {
            "text": text,
            "source": source_language,
            "target": target_language
        }

        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result.get("translated_text", text)
        else:
            raise Exception(f"Translation service error: {response.status_code}")

    def _translate_with_llm(self, text: str, source_language: str, target_language: str) -> str:
        """
        Translate text using the LLM.

        Args:
            text (str): Text to translate
            source_language (str): Source language code
            target_language (str): Target language code

        Returns:
            str: Translated text
        """
        # Get language names for better prompting
        source_name = self.get_language_name(source_language)
        target_name = self.get_language_name(target_language)

        # Prepare prompt
        prompt = f"Translate the following text from {source_name} to {target_name}. Maintain the original formatting and meaning as closely as possible.\n\nText to translate:\n{text}\n\nTranslation:"

        # Send to LLM
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.3
        }

        try:
            response = requests.post(COMPLETIONS_ENDPOINT, json=payload)

            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    translated_text = data['choices'][0].get("text", "").strip()
                    logger.info(f"Successfully translated text from {source_language} to {target_language}")
                    return translated_text
                else:
                    logger.error("No translation in LLM response")
                    return text
            else:
                logger.error(f"LLM service error: {response.status_code}")
                return text
        except Exception as e:
            logger.error(f"Error using LLM for translation: {e}")
            return text

    def translate_to_dialect(self, text: str, language_code: str, dialect: str) -> str:
        """
        Translate or adapt text to a specific dialect.

        Args:
            text (str): Text to adapt
            language_code (str): Base language code
            dialect (str): Target dialect code

        Returns:
            str: Dialect-adapted text
        """
        # Skip if no dialect specified
        if not dialect:
            return text

        dialect_key = f"{language_code}_{dialect}"

        # Check if we need dialect adaptation
        if dialect_key not in DIALECT_ADJUSTMENTS:
            return text

        # Use LLM to adapt to dialect
        prompt = ""

        # Create dialect-specific prompts
        if dialect_key == "ar_TN":
            prompt = f"Convert the following Modern Standard Arabic text to Tunisian Arabic dialect (Derja). Keep the same meaning but use Tunisian expressions and grammar:\n\nText:\n{text}\n\nTunisian dialect:"
        elif dialect_key == "ar_EG":
            prompt = f"Convert the following Modern Standard Arabic text to Egyptian Arabic dialect (Ammiya). Keep the same meaning but use Egyptian expressions and grammar:\n\nText:\n{text}\n\nEgyptian dialect:"
        elif dialect_key == "fr_TN":
            prompt = f"Convert the following standard French text to Tunisian French dialect. Use phrases and expressions common in Tunisia while keeping the same meaning:\n\nText:\n{text}\n\nTunisian French:"
        else:
            # Generic dialect adaptation
            prompt = f"Convert the following text in {self.get_language_name(language_code)} to {dialect} dialect. Keep the same meaning but use appropriate expressions and grammar:\n\nText:\n{text}\n\nDialect version:"

        # Send to LLM
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.4  # Slightly higher temperature for creativity in dialect
        }

        try:
            response = requests.post(COMPLETIONS_ENDPOINT, json=payload)

            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    dialect_text = data['choices'][0].get("text", "").strip()
                    logger.info(f"Successfully adapted text to {dialect_key} dialect")
                    return dialect_text
                else:
                    logger.error("No dialect adaptation in LLM response")
                    return text
            else:
                logger.error(f"LLM service error: {response.status_code}")
                return text
        except Exception as e:
            logger.error(f"Error using LLM for dialect adaptation: {e}")
            return text

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.

        Args:
            text (str): Text to analyze

        Returns:
            str: Detected language code
        """
        if not text or len(text.strip()) < 5:
            return self.default_language

        # Use LLM to detect language
        prompt = f"Identify the language of the following text. Respond with just the language code (e.g. 'en' for English, 'ar' for Arabic, 'fr' for French, etc.):\n\nText:\n{text}\n\nLanguage code:"

        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "max_tokens": 10,
            "temperature": 0.1
        }

        try:
            response = requests.post(COMPLETIONS_ENDPOINT, json=payload)

            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    lang_code = data['choices'][0].get("text", "").strip().lower()

                    # Extract just the language code if needed
                    lang_code = lang_code.split()[0] if " " in lang_code else lang_code

                    # Verify it's a supported language
                    if lang_code in SUPPORTED_LANGUAGES:
                        logger.info(f"Detected language: {lang_code}")
                        return lang_code
                    else:
                        logger.warning(f"Detected unsupported language: {lang_code}, falling back to {self.default_language}")
                        return self.default_language

            logger.error("Language detection failed")
            return self.default_language
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return self.default_language

    def format_recommendation(self,
                             plant_name: str,
                             growth_factors: Dict[str, str],
                             environment: Dict[str, str],
                             summary: str,
                             language_code: str,
                             dialect: str = None) -> str:
        """
        Format a plant care recommendation in the target language and dialect.

        Args:
            plant_name (str): Name of the plant
            growth_factors (dict): Growth factor information
            environment (dict): User's environment information
            summary (str): Summary text
            language_code (str): Target language code
            dialect (str, optional): Target dialect code

        Returns:
            str: Formatted recommendation
        """
        # Translate plant name if needed (for display purposes)
        display_plant_name = plant_name

        # Get recommendation intro text
        intro = self.get_text("recommendation_intro", language_code, dialect, plant=display_plant_name)

        # Format growth factors with localized titles
        factors_text = ""
        for factor, value in growth_factors.items():
            if value and value != "Not specified":
                title = self.get_growth_factor_title(factor, language_code)
                factors_text += f"**{title}**: {value}\n"

        # Format user environment and compatibility
        environment_text = ""
        if environment:
            environment_title = self.templates.get(language_code, {}).get("environment_title",
                                self.templates["en"].get("environment_title", "Your Environment"))

            environment_text = f"\n**{environment_title}**:\n"
            for factor, value in environment.items():
                if value:
                    factor_title = self.get_growth_factor_title(factor, language_code)
                    environment_text += f"**{factor_title}**: {value}\n"

        # Combine all parts
        recommendation = f"{intro}\n\n{summary}\n\n**{self.get_text('growth_factors_title', language_code, dialect)}**:\n{factors_text}"

        if environment_text:
            recommendation += f"\n{environment_text}"

        # Add adaptation tips if environment differs from optimal

        # Translate to dialect if needed
        if dialect:
            recommendation = self.translate_to_dialect(recommendation, language_code, dialect)

        return recommendation

    def get_plant_names_in_languages(self, base_plant_name: str, target_languages: List[str]) -> Dict[str, str]:
        """
        Get translations of a plant name in multiple languages.

        Args:
            base_plant_name (str): Base plant name (typically in English)
            target_languages (list): List of target language codes

        Returns:
            dict: Dictionary of translated plant names by language code
        """
        translations = {self.default_language: base_plant_name}

        for lang in target_languages:
            if lang == self.default_language:
                continue

            # Translate plant name
            prompt = f"Translate the plant name '{base_plant_name}' to {self.get_language_name(lang)}. Return ONLY the translated plant name, nothing else."

            payload = {
                "model": DEFAULT_MODEL,
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.2
            }

            try:
                response = requests.post(COMPLETIONS_ENDPOINT, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        translated_name = data['choices'][0].get("text", "").strip()
                        translations[lang] = translated_name
                    else:
                        translations[lang] = base_plant_name
                else:
                    translations[lang] = base_plant_name
            except Exception as e:
                logger.error(f"Error translating plant name to {lang}: {e}")
                translations[lang] = base_plant_name

        return translations

# For testing
if __name__ == "__main__":
    language_manager = LanguageManager()

    # Test getting text in different languages
    print("Testing text retrieval:")
    print(f"English greeting: {language_manager.get_text('greeting', 'en')}")
    print(f"Arabic greeting: {language_manager.get_text('greeting', 'ar')}")
    print(f"Tunisian Arabic greeting: {language_manager.get_text('greeting', 'ar', 'TN')}")

    # Test translation
    test_text = input("\nEnter text to translate: ")
    source_lang = input("Source language code (e.g., en, ar, fr): ")
    target_lang = input("Target language code: ")

    translated = language_manager.translate_text(test_text, source_lang, target_lang)
    print(f"\nTranslated text: {translated}")

    # Test dialect adaptation if relevant
    if target_lang in ["ar", "fr"]:
        dialect = input(f"\nEnter dialect for {target_lang} (e.g., TN for Tunisian): ")
        if dialect:
            dialect_text = language_manager.translate_to_dialect(translated, target_lang, dialect)
            print(f"\nDialect-adapted text: {dialect_text}")