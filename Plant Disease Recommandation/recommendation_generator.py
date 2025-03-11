"""
Recommendation generator for creating personalized treatment recommendations.
Uses LM Studio API or a fallback mechanism to generate recommendations.
"""

import requests
import re
from typing import Dict, Any, List, Optional
import os

from config import (DEFAULT_LM_STUDIO_URL, LM_SYSTEM_PROMPT,
                    ORGANIC_TREATMENTS, CONVENTIONAL_TREATMENTS, MODEL_TOKEN_LIMITS)
import utils


class RecommendationGenerator:
    def __init__(self, lm_studio_url: str = DEFAULT_LM_STUDIO_URL):
        """
        Initialize the recommendation generator.

        Args:
            lm_studio_url: URL of the LM Studio API server
        """
        self.lm_studio_url = lm_studio_url

        # Test LM Studio connection
        try:
            self._test_lm_studio_connection()
            print("✅ Successfully connected to LM Studio")
            self.lm_studio_available = True
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to LM Studio at {lm_studio_url}")
            print(f"   Error: {e}")
            print("   Make sure LM Studio is running and the API server is enabled.")
            print("   The system will still work, but will use mock responses instead.")
            self.lm_studio_available = False

    def _test_lm_studio_connection(self) -> bool:
        """
        Test the connection to LM Studio.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.lm_studio_url}/models")
            if response.status_code == 200:
                models = response.json()
                if "data" in models and len(models["data"]) > 0:
                    print(f"Available models in LM Studio: {', '.join([m['id'] for m in models['data']])}")
                return True
            return False
        except Exception:
            return False

    def generate_recommendation_prompt(self, disease_info: Dict[str, Any],
                                       user_details: Dict[str, str],
                                       disease_name: Optional[str] = None,
                                       max_tokens: int = 2000) -> str:
        """
        Generate a prompt for the LLM to create personalized recommendations.

        Args:
            disease_info: Dictionary containing disease information
            user_details: Dictionary containing user-provided details
            disease_name: Specific disease to focus on (optional)
            max_tokens: Maximum tokens for context

        Returns:
            Prompt for the LLM
        """
        plant_name = disease_info["plant"]

        # Create the base prompt
        prompt = f"""As a plant pathologist, create a personalized recommendation for treating {plant_name} with these disease details:

"""

        # Add disease information
        for disease in disease_info["diseases"]:
            prompt += f"DISEASE: {disease['name']}\n\n"

            if disease.get("symptoms"):
                prompt += "Key Symptoms:\n"
                for symptom in disease["symptoms"]:
                    prompt += f"- {symptom}\n"
                prompt += "\n"

            if disease.get("causes"):
                prompt += "Main Causes:\n"
                for cause in disease["causes"]:
                    prompt += f"- {cause}\n"
                prompt += "\n"

            if disease.get("treatments"):
                prompt += "Effective Treatments:\n"
                for treatment in disease["treatments"]:
                    prompt += f"- {treatment}\n"
                prompt += "\n"

            if disease.get("prevention"):
                prompt += "Key Prevention Methods:\n"
                for prevention in disease["prevention"]:
                    prompt += f"- {prevention}\n"
                prompt += "\n"

        # Add user details
        prompt += "GARDENER DETAILS:\n"
        prompt += f"- Location/Climate: {user_details.get('location', 'Unknown')}\n"
        prompt += f"- Growing Environment: {user_details.get('environment', 'Unknown')}\n"
        prompt += f"- Prefers Organic Solutions: {user_details.get('organic_preferred', False)}\n"
        prompt += f"- Disease Severity: {user_details.get('severity', 'Unknown')}\n"
        prompt += f"- Other Plants Nearby: {user_details.get('other_plants', 'Unknown')}\n"
        prompt += f"- Previous Treatments Applied: {user_details.get('previous_treatments', 'None')}\n\n"

        # Add instructions for the response
        prompt += """Create a comprehensive recommendation including:
1. DIAGNOSIS CONFIRMATION: Confirm if symptoms match the disease
2. IMMEDIATE TREATMENT: Steps to take now (favor organic if preferred)
3. LONG-TERM MANAGEMENT: Strategies for prevention and control
4. CULTURAL PRACTICES: Watering, pruning, or other care changes
5. ADDITIONAL CONSIDERATIONS: Other relevant factors

Keep your response practical with emphasis on the gardener's preferences.
"""

        # Estimate token count
        approx_token_count = utils.estimate_token_count(prompt)
        print(f"Approximate token count of prompt: {approx_token_count} tokens")

        return prompt

    def call_lm_studio(self, prompt: str) -> str:
        """
        Call the LM Studio API to generate a recommendation.

        Args:
            prompt: The prompt to send to LM Studio

        Returns:
            The generated recommendation
        """
        try:
            headers = {
                "Content-Type": "application/json"
            }

            data = {
                "messages": [
                    {"role": "system", "content": LM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1500,
                "stream": False
            }

            response = requests.post(
                f"{self.lm_studio_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60  # Local LLMs might take longer to respond
            )

            # Check for errors
            if response.status_code != 200:
                print(f"Error calling LM Studio API: {response.status_code}")
                print(f"Response: {response.text}")
                return self._generate_fallback_recommendation(prompt)

            # Extract the response
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                print("Unexpected response format from LM Studio")
                return self._generate_fallback_recommendation(prompt)

        except Exception as e:
            print(f"Error calling LM Studio API: {e}")
            return self._generate_fallback_recommendation(prompt)

    def _generate_fallback_recommendation(self, prompt: str) -> str:
        """
        Generate a fallback recommendation using a rule-based approach.

        Args:
            prompt: The original prompt

        Returns:
            A generated recommendation
        """
        # Extract key information from the prompt
        plant_match = re.search(r"treating\s+(\w+)", prompt)
        plant_name = plant_match.group(1) if plant_match else "your plant"

        disease_match = re.search(r"DISEASE:\s*(.+?)$", prompt, re.MULTILINE)
        disease_name = disease_match.group(1) if disease_match else "the disease"

        organic_preferred = "Prefers Organic Solutions: True" in prompt
        severity_match = re.search(r"Disease Severity:\s*(\w+)", prompt)
        severity = severity_match.group(1).lower() if severity_match else "unknown"

        # Extract some symptoms if available
        symptoms_text = ""
        symptoms_block = re.search(r"Key Symptoms:\n((?:-.*\n)+)", prompt)
        if symptoms_block:
            symptoms = re.findall(r"- (.*)\n", symptoms_block.group(1))
            if symptoms:
                symptoms_text = ", ".join(symptoms[:2])

        # Extract some treatments if available
        treatments = []
        treatments_block = re.search(r"Effective Treatments:\n((?:-.*\n)+)", prompt)
        if treatments_block:
            treatments = re.findall(r"- (.*)\n", treatments_block.group(1))

        # Generate appropriate treatments based on preferences and severity
        organic_treatments = ORGANIC_TREATMENTS
        conventional_treatments = CONVENTIONAL_TREATMENTS

        # Select treatments based on preference and add any from the original text
        if organic_preferred:
            selected_treatments = organic_treatments
        else:
            selected_treatments = conventional_treatments

        # Add treatments from the original text if they match the preferences
        for treatment in treatments:
            is_organic = any(
                term in treatment.lower() for term in ["organic", "natural", "neem", "soap", "compost", "baking soda"])
            if organic_preferred and is_organic:
                if treatment not in selected_treatments:
                    selected_treatments.append(treatment)
            elif not organic_preferred and not is_organic:
                if treatment not in selected_treatments:
                    selected_treatments.append(treatment)

        # Generate severity-appropriate advice
        if severity == "mild":
            treatment_advice = f"{selected_treatments[0]}"
            if len(selected_treatments) > 1:
                treatment_advice += f" If no improvement within 10 days, try {selected_treatments[1]}"
        elif severity == "severe":
            treatment_advice = f"Immediately {selected_treatments[0]} Then follow up with {selected_treatments[1] if len(selected_treatments) > 1 else 'a second application'} after 7 days."
        else:  # moderate or unknown
            treatment_advice = f"{selected_treatments[0]} Follow up with {selected_treatments[1] if len(selected_treatments) > 1 else 'a second application'} after 10-14 days."

        # Generate a response based on the extracted information
        response = f"""# Treatment Recommendation for {plant_name} with {disease_name}

## DIAGNOSIS CONFIRMATION
Based on the symptoms described{f" ({symptoms_text})" if symptoms_text else ""}, this appears to be {disease_name}. The symptoms you're observing are consistent with this disease, particularly at a {severity} stage.

## IMMEDIATE TREATMENT
{treatment_advice}

## LONG-TERM MANAGEMENT
1. Remove and dispose of all infected plant material
2. Maintain proper tree spacing for air circulation
3. Apply dormant oil spray in late winter
4. Consider resistant varieties for future plantings

## CULTURAL PRACTICES
- Water early in the day at the base of plants to keep foliage dry
- Prune regularly to improve airflow and remove diseased branches
- {"Use organic fertilizers that promote slower, steadier growth" if organic_preferred else "Avoid excess nitrogen fertilization which promotes susceptible new growth"}

## ADDITIONAL CONSIDERATIONS
- Monitor nearby plants closely as the disease can spread
- Sanitize all pruning tools with 70% alcohol between cuts
- Document treatment effectiveness for future reference

"""

        return response

    def get_model_token_limit(self) -> int:
        """
        Try to determine the token limit of the loaded model in LM Studio.

        Returns:
            Estimated token limit for the model
        """
        try:
            response = requests.get(f"{self.lm_studio_url}/models")
            if response.status_code == 200:
                models = response.json()
                if "data" in models and len(models["data"]) > 0:
                    model_id = models["data"][0]["id"].lower()

                    # Estimate based on model name
                    for key, limit in MODEL_TOKEN_LIMITS.items():
                        if key in model_id:
                            return limit

            # Conservative default
            return 2048
        except:
            # Very conservative fallback
            return 2048

    def generate_recommendation(self, disease_info: Dict[str, Any], user_details: Dict[str, str],
                                disease_name: Optional[str] = None) -> str:
        """
        Generate a personalized recommendation for treating a plant disease.

        Args:
            disease_info: Dictionary containing disease information
            user_details: Dictionary containing user-provided details
            disease_name: Specific disease to focus on (optional)

        Returns:
            The generated recommendation
        """
        try:
            # Determine the model's token limit
            model_token_limit = self.get_model_token_limit()

            # Generate prompt for the LLM
            prompt = self.generate_recommendation_prompt(
                disease_info,
                user_details,
                disease_name,
                max_tokens=model_token_limit - 1000  # Leave room for the response
            )

            print("\nGenerating personalized recommendation...")

            # Call the LLM to generate the recommendation
            if self.lm_studio_available:
                recommendation = self.call_lm_studio(prompt)
            else:
                recommendation = self._generate_fallback_recommendation(prompt)

            return recommendation

        except Exception as e:
            return f"An error occurred while generating the recommendation: {str(e)}"


# Simple test function
def test_generator():
    from data_processor import DataProcessor
    from context_analyzer import ContextAnalyzer

    processor = DataProcessor()

    try:
        # Try to load existing data
        info = processor.load_disease_info("tomato")

        # Define mock user query
        user_query = {
            "location": "Zone 7b",
            "environment": "outdoor garden",
            "organic_preferred": True,
            "severity": "moderate",
            "other_plants": "yes, tomatoes and peppers nearby",
            "previous_treatments": "neem oil"
        }

        analyzer = ContextAnalyzer()
        relevant = analyzer.extract_relevant_context(info, user_query)

        generator = RecommendationGenerator()
        recommendation = generator.generate_recommendation(relevant, user_query)

        print("\n" + "=" * 80)
        print("PERSONALIZED TREATMENT RECOMMENDATION")
        print("=" * 80 + "\n")
        print(recommendation)

    except FileNotFoundError:
        print("No existing data found. Run the data processor first.")


if __name__ == "__main__":
    test_generator()