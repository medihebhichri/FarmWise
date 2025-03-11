"""
environmental_analyzer.py
Analyzes user's environmental conditions for plant growing and compares
with optimal growing conditions to provide personalized recommendations.
"""

import os
import logging
import json
import re
import requests
from typing import Dict, Any, List, Tuple, Optional

# Import configuration
from config import (
    LOGS_DIR, LLM_SERVER_URL, COMPLETIONS_ENDPOINT, DEFAULT_MODEL,
    USER_ENV_FACTORS, GROWTH_FACTOR_KEYS
)

# Set up logging
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "environmental_analyzer.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnvironmentalAnalyzer")


class EnvironmentalAnalyzer:
    """Analyzes and compares growing environments for plants."""

    def __init__(self):
        """Initialize the environmental analyzer."""
        # Common units and their normalized values
        self.temperature_units = {
            "c": "Celsius",
            "f": "Fahrenheit",
            "celsius": "Celsius",
            "fahrenheit": "Fahrenheit",
            "°c": "Celsius",
            "°f": "Fahrenheit"
        }

        self.humidity_units = {
            "%": "percent",
            "percent": "percent",
            "percentage": "percent"
        }

    def parse_user_environment(self, user_input: str) -> Dict[str, Any]:
        """
        Parse user's description of their growing environment.

        Args:
            user_input (str): User's description of their environment

        Returns:
            dict: Structured environmental factors
        """
        # Use LLM to extract environmental factors from user input
        prompt = (
            "Extract the following environmental factors from the user's description "
            "of their growing environment. Return a JSON object with these keys: "
            "temperature, humidity, soil_humidity, wind_exposure, rainfall, sunlight_hours, location_type, season.\n\n"
            f"User description: {user_input}\n\n"
            "For any factor not mentioned, use the value 'Not specified'. "
            "For temperature, include the unit (C or F) if mentioned. "
            "For location_type, use 'indoor', 'outdoor', or 'greenhouse'. "
            "Return ONLY a valid JSON object."
        )

        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.2
        }

        try:
            response = requests.post(COMPLETIONS_ENDPOINT, json=payload)
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    result_text = data['choices'][0].get("text", "").strip()

                    try:
                        # Try to parse JSON response
                        env_factors = json.loads(result_text)

                        # Ensure all required keys are present
                        parsed_environment = {}
                        for factor in USER_ENV_FACTORS:
                            parsed_environment[factor] = env_factors.get(factor, "Not specified")

                        logger.info(f"Successfully parsed environment factors")
                        return parsed_environment

                    except json.JSONDecodeError:
                        # Try to extract JSON from text if direct parsing fails
                        match = re.search(r'\{.*\}', result_text, re.DOTALL)
                        if match:
                            try:
                                env_factors = json.loads(match.group(0))

                                # Ensure all required keys are present
                                parsed_environment = {}
                                for factor in USER_ENV_FACTORS:
                                    parsed_environment[factor] = env_factors.get(factor, "Not specified")

                                logger.info(f"Successfully parsed environment factors from extracted JSON")
                                return parsed_environment
                            except Exception:
                                pass

                # If we get here, extraction failed
                logger.warning("Failed to extract environment factors")
                return {factor: "Not specified" for factor in USER_ENV_FACTORS}
            else:
                logger.error(f"LLM server error: {response.status_code}")
                return {factor: "Not specified" for factor in USER_ENV_FACTORS}
        except Exception as e:
            logger.error(f"Exception in environment parsing: {e}")
            return {factor: "Not specified" for factor in USER_ENV_FACTORS}

    def normalize_temperature(self, temp_str: str) -> Tuple[Optional[float], Optional[str], str]:
        """
        Normalize temperature values for comparison.

        Args:
            temp_str (str): Temperature string (e.g., "75°F", "25C")

        Returns:
            tuple: (normalized value in Celsius, original unit, normalized representation)
        """
        if not temp_str or temp_str == "Not specified":
            return None, None, temp_str

        # Extract numeric value and unit
        match = re.search(r'(-?\d+(?:\.\d+)?)\s*([CF°])', temp_str, re.IGNORECASE)
        if not match:
            # Try to find just a number
            num_match = re.search(r'(-?\d+(?:\.\d+)?)', temp_str)
            if num_match:
                # Assume Celsius if no unit specified
                value = float(num_match.group(1))
                return value, "C", f"{value}°C"
            return None, None, temp_str

        value = float(match.group(1))
        unit = match.group(2).upper()

        # Normalize to Celsius
        if unit in ['F', '°F']:
            celsius = (value - 32) * 5 / 9
            return celsius, "F", f"{value}°F ({celsius:.1f}°C)"
        else:
            return value, "C", f"{value}°C"

    def normalize_humidity(self, humidity_str: str) -> Tuple[Optional[float], str]:
        """
        Normalize humidity values for comparison.

        Args:
            humidity_str (str): Humidity string (e.g., "65%", "high")

        Returns:
            tuple: (normalized value as percentage, normalized representation)
        """
        if not humidity_str or humidity_str == "Not specified":
            return None, humidity_str

        # Extract numeric value if present
        match = re.search(r'(\d+(?:\.\d+)?)\s*(%|percent)', humidity_str, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return value, f"{value}%"

        # Handle qualitative descriptions
        humidity_map = {
            "very low": 10,
            "low": 30,
            "medium": 50,
            "moderate": 50,
            "high": 70,
            "very high": 90
        }

        for desc, value in humidity_map.items():
            if desc in humidity_str.lower():
                return value, f"{desc} (approximately {value}%)"

        return None, humidity_str

    def compare_environments(self, optimal_factors: Dict[str, str], user_environment: Dict[str, str]) -> Dict[str, Any]:
        """
        Compare optimal growing conditions with user's environment.

        Args:
            optimal_factors (dict): Optimal growth factors for the plant
            user_environment (dict): User's environment factors

        Returns:
            dict: Comparison results with recommendations
        """
        comparison = {
            "compatibility": {},
            "recommendations": [],
            "overall_score": 0,
            "factors_analyzed": 0
        }

        # Map growth factors to user environment factors
        factor_mapping = {
            "temperature": "temperature",
            "humidity": "humidity",
            "soil_type": "soil_humidity",
            "light_exposure": "sunlight_hours",
            "water_requirements": "rainfall"
        }

        total_score = 0
        factors_analyzed = 0

        # Compare each factor
        for plant_factor, env_factor in factor_mapping.items():
            if (plant_factor in optimal_factors and
                    optimal_factors[plant_factor] != "Not specified" and
                    env_factor in user_environment and
                    user_environment[env_factor] != "Not specified"):

                optimal = optimal_factors[plant_factor]
                actual = user_environment[env_factor]

                # Analyze temperature
                if plant_factor == "temperature":
                    opt_temp, opt_unit, opt_normalized = self.normalize_temperature(optimal)
                    actual_temp, actual_unit, actual_normalized = self.normalize_temperature(actual)

                    if opt_temp is not None and actual_temp is not None:
                        temp_diff = abs(opt_temp - actual_temp)

                        if temp_diff <= 3:
                            compatibility = "Excellent"
                            score = 1.0
                        elif temp_diff <= 7:
                            compatibility = "Good"
                            score = 0.7
                        elif temp_diff <= 12:
                            compatibility = "Moderate"
                            score = 0.4
                        else:
                            compatibility = "Poor"
                            score = 0.1

                        recommendation = ""
                        if actual_temp < opt_temp:
                            recommendation = f"Your temperature is lower than optimal. Consider increasing temperature or providing additional heating."
                        elif actual_temp > opt_temp:
                            recommendation = f"Your temperature is higher than optimal. Consider shade or cooling methods."

                        comparison["compatibility"][plant_factor] = {
                            "optimal": opt_normalized,
                            "actual": actual_normalized,
                            "compatibility": compatibility,
                            "score": score
                        }

                        if recommendation:
                            comparison["recommendations"].append(recommendation)

                        total_score += score
                        factors_analyzed += 1

                # Analyze humidity
                elif plant_factor == "humidity":
                    opt_humidity, opt_normalized = self.normalize_humidity(optimal)
                    actual_humidity, actual_normalized = self.normalize_humidity(actual)

                    if opt_humidity is not None and actual_humidity is not None:
                        humidity_diff = abs(opt_humidity - actual_humidity)

                        if humidity_diff <= 10:
                            compatibility = "Excellent"
                            score = 1.0
                        elif humidity_diff <= 20:
                            compatibility = "Good"
                            score = 0.7
                        elif humidity_diff <= 30:
                            compatibility = "Moderate"
                            score = 0.4
                        else:
                            compatibility = "Poor"
                            score = 0.1

                        recommendation = ""
                        if actual_humidity < opt_humidity:
                            recommendation = f"Your humidity is lower than optimal. Consider using a humidifier or misting the plant regularly."
                        elif actual_humidity > opt_humidity:
                            recommendation = f"Your humidity is higher than optimal. Consider improving air circulation or using a dehumidifier."

                        comparison["compatibility"][plant_factor] = {
                            "optimal": opt_normalized,
                            "actual": actual_normalized,
                            "compatibility": compatibility,
                            "score": score
                        }

                        if recommendation:
                            comparison["recommendations"].append(recommendation)

                        total_score += score
                        factors_analyzed += 1

                # Text-based comparisons for other factors
                else:
                    # Use LLM to compare text descriptions
                    prompt = (
                        f"Compare these growing conditions for a plant:\n"
                        f"Optimal {plant_factor}: {optimal}\n"
                        f"Actual {env_factor}: {actual}\n\n"
                        f"Rate the compatibility as 'Excellent', 'Good', 'Moderate', or 'Poor'. "
                        f"Also provide a score between 0 and 1, and a brief recommendation if needed. "
                        f"Return as JSON with keys: compatibility, score, recommendation"
                    )

                    payload = {
                        "model": DEFAULT_MODEL,
                        "prompt": prompt,
                        "max_tokens": 200,
                        "temperature": 0.3
                    }

                    try:
                        response = requests.post(COMPLETIONS_ENDPOINT, json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            if 'choices' in data and len(data['choices']) > 0:
                                result_text = data['choices'][0].get("text", "").strip()

                                try:
                                    # Parse JSON response
                                    factor_comparison = json.loads(result_text)

                                    compatibility = factor_comparison.get("compatibility", "Unknown")
                                    score = float(factor_comparison.get("score", 0.5))
                                    recommendation = factor_comparison.get("recommendation", "")

                                    comparison["compatibility"][plant_factor] = {
                                        "optimal": optimal,
                                        "actual": actual,
                                        "compatibility": compatibility,
                                        "score": score
                                    }

                                    if recommendation:
                                        comparison["recommendations"].append(recommendation)

                                    total_score += score
                                    factors_analyzed += 1

                                except (json.JSONDecodeError, ValueError):
                                    # Fall back to simple comparison
                                    comparison["compatibility"][plant_factor] = {
                                        "optimal": optimal,
                                        "actual": actual,
                                        "compatibility": "Unknown",
                                        "score": 0.5
                                    }

                                    total_score += 0.5
                                    factors_analyzed += 1
                    except Exception as e:
                        logger.error(f"Error comparing {plant_factor}: {e}")

        # Calculate overall score
        if factors_analyzed > 0:
            comparison["overall_score"] = total_score / factors_analyzed
            comparison["factors_analyzed"] = factors_analyzed

            # Add overall compatibility rating
            score = comparison["overall_score"]
            if score >= 0.8:
                comparison["overall_compatibility"] = "Excellent"
            elif score >= 0.6:
                comparison["overall_compatibility"] = "Good"
            elif score >= 0.4:
                comparison["overall_compatibility"] = "Moderate"
            else:
                comparison["overall_compatibility"] = "Poor"
        else:
            comparison["overall_score"] = 0
            comparison["factors_analyzed"] = 0
            comparison["overall_compatibility"] = "Unknown"

        return comparison

    def generate_adaptation_tips(self, plant_name: str, comparison_results: Dict[str, Any],
                                 language_code: str = "en") -> str:
        """
        Generate adaptation tips based on environmental comparison.

        Args:
            plant_name (str): Name of the plant
            comparison_results (dict): Comparison results
            language_code (str): Language code for the response

        Returns:
            str: Adaptation tips
        """
        overall_score = comparison_results.get("overall_score", 0)
        recommendations = comparison_results.get("recommendations", [])
        compatibility = comparison_results.get("compatibility", {})

        # Prepare context for LLM
        context = f"Plant: {plant_name}\n"
        context += f"Overall compatibility score: {overall_score:.2f}\n"
        context += "Compatibility by factor:\n"

        for factor, details in compatibility.items():
            context += f"- {factor}: {details.get('compatibility', 'Unknown')} (Score: {details.get('score', 0):.2f})\n"
            context += f"  Optimal: {details.get('optimal', 'Unknown')}\n"
            context += f"  Actual: {details.get('actual', 'Unknown')}\n"

        context += "\nRecommendations:\n"
        for rec in recommendations:
            context += f"- {rec}\n"

        # Generate adaptation tips using LLM
        language_prompt = {
            "en": f"Based on the environmental comparison for growing {plant_name}, provide practical adaptation tips to help the user improve their growing conditions. Focus on actionable advice for factors with low compatibility scores.",
            "ar": f"بناءً على مقارنة البيئة لزراعة {plant_name}، قدم نصائح عملية للتكيف لمساعدة المستخدم على تحسين ظروف النمو. ركز على نصائح قابلة للتنفيذ للعوامل ذات درجات التوافق المنخفضة.",
            "fr": f"En fonction de la comparaison environnementale pour la culture de {plant_name}, fournissez des conseils d'adaptation pratiques pour aider l'utilisateur à améliorer ses conditions de culture. Concentrez-vous sur des conseils pratiques pour les facteurs ayant des scores de compatibilité faibles."
        }

        # Default to English if language not supported
        prompt = language_prompt.get(language_code, language_prompt["en"])

        full_prompt = f"{prompt}\n\nEnvironmental comparison:\n{context}\n\nAdaptation tips:"

        payload = {
            "model": DEFAULT_MODEL,
            "prompt": full_prompt,
            "max_tokens": 500,
            "temperature": 0.4
        }

        try:
            response = requests.post(COMPLETIONS_ENDPOINT, json=payload)
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    tips = data['choices'][0].get("text", "").strip()
                    return tips

            # If we get here, something went wrong
            logger.error("Failed to generate adaptation tips")
            return "No adaptation tips available."
        except Exception as e:
            logger.error(f"Error generating adaptation tips: {e}")
            return "Error generating adaptation tips."

    def get_environment_summary(self, user_environment: Dict[str, str], language_code: str = "en") -> str:
        """
        Generate a human-readable summary of the user's environment.

        Args:
            user_environment (dict): User's environment factors
            language_code (str): Language code for the response

        Returns:
            str: Formatted environment summary
        """
        # Language-specific templates
        templates = {
            "en": {
                "title": "Your Growing Environment",
                "temperature": "Temperature",
                "humidity": "Humidity",
                "soil_humidity": "Soil Moisture",
                "wind_exposure": "Wind Exposure",
                "rainfall": "Rainfall",
                "sunlight_hours": "Sunlight",
                "location_type": "Location Type",
                "season": "Season",
                "not_specified": "Not specified"
            },
            "ar": {
                "title": "بيئة النمو الخاصة بك",
                "temperature": "درجة الحرارة",
                "humidity": "الرطوبة",
                "soil_humidity": "رطوبة التربة",
                "wind_exposure": "التعرض للرياح",
                "rainfall": "هطول الأمطار",
                "sunlight_hours": "أشعة الشمس",
                "location_type": "نوع الموقع",
                "season": "الموسم",
                "not_specified": "غير محدد"
            },
            "fr": {
                "title": "Votre Environnement de Culture",
                "temperature": "Température",
                "humidity": "Humidité",
                "soil_humidity": "Humidité du Sol",
                "wind_exposure": "Exposition au Vent",
                "rainfall": "Précipitations",
                "sunlight_hours": "Ensoleillement",
                "location_type": "Type d'Emplacement",
                "season": "Saison",
                "not_specified": "Non spécifié"
            }
        }

        # Use English as fallback
        lang_template = templates.get(language_code, templates["en"])

        summary = f"**{lang_template['title']}**\n\n"

        for factor, value in user_environment.items():
            if factor in lang_template:
                factor_name = lang_template[factor]
                if value and value != "Not specified":
                    summary += f"- **{factor_name}**: {value}\n"
                else:
                    summary += f"- **{factor_name}**: {lang_template['not_specified']}\n"

        return summary

    def is_environment_suitable(self, plant_name: str, user_environment: Dict[str, str],
                                optimal_factors: Dict[str, str]) -> Dict[str, Any]:
        """
        Determine if user's environment is suitable for growing the plant.

        Args:
            plant_name (str): Name of the plant
            user_environment (dict): User's environment factors
            optimal_factors (dict): Optimal growth factors for the plant

        Returns:
            dict: Suitability assessment with score and recommendations
        """
        # Compare environments
        comparison = self.compare_environments(optimal_factors, user_environment)

        # Generate adaptation tips
        adaptation_tips = self.generate_adaptation_tips(plant_name, comparison)

        # Prepare overall assessment
        assessment = {
            "plant_name": plant_name,
            "overall_score": comparison["overall_score"],
            "overall_compatibility": comparison["overall_compatibility"],
            "factor_comparison": comparison["compatibility"],
            "adaptation_tips": adaptation_tips,
            "recommendations": comparison["recommendations"]
        }

        return assessment


# For testing
if __name__ == "__main__":
    analyzer = EnvironmentalAnalyzer()

    # Test parsing user environment
    test_input = input("Describe your growing environment: ")
    parsed_env = analyzer.parse_user_environment(test_input)

    print("\nParsed Environment:")
    for factor, value in parsed_env.items():
        print(f"- {factor}: {value}")

    # Test temperature normalization
    if parsed_env["temperature"] != "Not specified":
        norm_temp, unit, display = analyzer.normalize_temperature(parsed_env["temperature"])
        print(f"\nNormalized temperature: {norm_temp}°C (Display: {display})")

    # Test humidity normalization
    if parsed_env["humidity"] != "Not specified":
        norm_humidity, display = analyzer.normalize_humidity(parsed_env["humidity"])
        print(f"Normalized humidity: {norm_humidity}% (Display: {display})")

    # Test with sample optimal factors
    sample_optimal = {
        "temperature": "22-25°C",
        "humidity": "60-70%",
        "soil_type": "Well-draining, rich in organic matter",
        "light_exposure": "Bright indirect light, 6-8 hours daily",
        "water_requirements": "Keep soil moist but not soggy"
    }

    plant_name = input("\nEnter a plant name: ")
    comparison = analyzer.compare_environments(sample_optimal, parsed_env)

    print("\nEnvironment Comparison Results:")
    print(f"Overall score: {comparison['overall_score']:.2f}")
    print(f"Factors analyzed: {comparison['factors_analyzed']}")

    print("\nFactor Compatibility:")
    for factor, details in comparison["compatibility"].items():
        print(f"- {factor}: {details['compatibility']} (Score: {details['score']:.2f})")
        print(f"  Optimal: {details['optimal']}")
        print(f"  Actual: {details['actual']}")

    print("\nRecommendations:")
    for rec in comparison["recommendations"]:
        print(f"- {rec}")

    # Test adaptation tips
    tips = analyzer.generate_adaptation_tips(plant_name, comparison)
    print("\nAdaptation Tips:")
    print(tips)