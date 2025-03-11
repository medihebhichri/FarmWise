import argparse
import json
import os
import re
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


class CAGProcessor:
    """
    Contextual Answer Generation processor for plant disease information.
    This class handles semantic understanding and context-aware extraction of information.
    """

    def __init__(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000
            )
            self.sklearn_available = True
        except ImportError:
            print("Warning: scikit-learn not available. Using basic text matching instead.")
            self.sklearn_available = False

    def extract_relevant_context(self, disease_info: Dict[str, Any], user_query: Dict[str, str],
                                 max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Extract the most relevant information based on the user's situation and query.
        Uses contextual relevance scoring rather than simple truncation.

        Args:
            disease_info: Dictionary containing disease information
            user_query: Dictionary with user details and implicit query
            max_tokens: Maximum tokens to include in the result

        Returns:
            Dictionary with the most contextually relevant information
        """
        # Create a consolidated context from the user's situation
        user_context = self._create_user_context(user_query)

        # Extract diseases and create relevance-ranked content
        relevant_info = {
            "plant": disease_info["plant"],
            "diseases": [],
            "sources": disease_info.get("sources", [])[:2]  # Limit sources to save tokens
        }

        # Process each disease and find the most relevant information
        for disease in disease_info.get("diseases", []):
            processed_disease = {
                "name": disease.get("name", "Unknown Disease"),
                "symptoms": [],
                "causes": [],
                "treatments": [],
                "prevention": []
            }

            # Get the most relevant items for each category based on user context
            for category in ["symptoms", "causes", "treatments", "prevention"]:
                items = disease.get(category, [])
                if not items:
                    continue

                # Calculate relevance scores for each item
                if self.sklearn_available:
                    scores = self._calculate_relevance_scores(items, user_context)
                else:
                    scores = self._calculate_relevance_basic(items, user_context)

                # Select the most relevant items, with token budget in mind
                token_budget = self._get_token_budget(category, max_tokens)
                selected_items = self._select_top_items(items, scores, token_budget)

                processed_disease[category] = selected_items

            relevant_info["diseases"].append(processed_disease)

        # Further filter if we still have too many tokens
        estimated_tokens = self._estimate_token_count(relevant_info)
        if estimated_tokens > max_tokens:
            # Apply more aggressive filtering
            return self._reduce_token_count(relevant_info, max_tokens)

        return relevant_info

    def _create_user_context(self, user_query: Dict[str, str]) -> str:
        """
        Create a consolidated context string from user information.
        This represents the implicit query/need of the user.
        """
        context_parts = []

        # Map severity to contextual importance
        severity_map = {
            "mild": "early stage treatment mild case",
            "moderate": "treatment for established infection moderate severity",
            "severe": "urgent treatment severe infection advanced stage"
        }

        # Add location context
        if "location" in user_query:
            context_parts.append(f"growing in {user_query['location']}")

        # Add environment context
        if "environment" in user_query:
            context_parts.append(f"{user_query['environment']} growing conditions")

        # Add treatment preference context
        if user_query.get("organic_preferred"):
            context_parts.append("organic treatment natural solution non-chemical")
        else:
            context_parts.append("effective treatment solution")

        # Add severity context
        severity = user_query.get("severity", "unknown").lower()
        if severity in severity_map:
            context_parts.append(severity_map[severity])

        # Add context about other plants
        if "other_plants" in user_query and "yes" in user_query["other_plants"].lower():
            context_parts.append("prevent spread to nearby plants containment")

        # Add previous treatment context
        if "previous_treatments" in user_query and user_query["previous_treatments"].lower() not in ["no", "none", ""]:
            context_parts.append(f"after trying {user_query['previous_treatments']} alternative treatment")

        return " ".join(context_parts)

    def _calculate_relevance_scores(self, items: List[str], context: str) -> np.ndarray:
        """
        Calculate relevance scores for each item based on the user context
        using scikit-learn's TF-IDF and cosine similarity.

        Args:
            items: List of text items to score
            context: User context string

        Returns:
            Array of relevance scores
        """
        # Add the context as the last document
        all_docs = items + [context]

        try:
            from sklearn.metrics.pairwise import cosine_similarity
            # Fit and transform the documents
            tfidf_matrix = self.vectorizer.fit_transform(all_docs)

            # The last row is the context vector
            context_vector = tfidf_matrix[-1]

            # Calculate cosine similarity between each item and the context
            similarities = cosine_similarity(tfidf_matrix[:-1], context_vector)

            # Return the flattened array of similarities
            return similarities.flatten()
        except:
            # Fallback if vectorization fails (e.g., not enough data)
            return np.ones(len(items))

    def _calculate_relevance_basic(self, items: List[str], context: str) -> List[float]:
        """
        Calculate relevance scores using basic text matching when scikit-learn is not available.

        Args:
            items: List of text items to score
            context: User context string

        Returns:
            List of relevance scores
        """
        # Split the context into keywords
        context_words = set(context.lower().split())

        # Calculate scores based on word overlap
        scores = []
        for item in items:
            item_words = set(item.lower().split())
            # Score is the number of overlapping words
            overlap = len(context_words.intersection(item_words))
            # Normalize by the length of the item
            score = overlap / max(1, len(item_words))
            scores.append(score)

        return scores

    def _select_top_items(self, items: List[str], scores: List[float], token_budget: int) -> List[str]:
        """
        Select the top items based on relevance scores and token budget.

        Args:
            items: List of text items
            scores: Relevance scores for each item
            token_budget: Maximum tokens to use

        Returns:
            List of selected items
        """
        # Create a list of (item, score) tuples
        item_scores = list(zip(items, scores))

        # Sort by score in descending order
        sorted_items = sorted(item_scores, key=lambda x: x[1], reverse=True)

        # Select items until we hit the token budget
        selected = []
        tokens_used = 0

        for item, _ in sorted_items:
            item_tokens = len(item.split())
            if tokens_used + item_tokens <= token_budget:
                selected.append(item)
                tokens_used += item_tokens
            else:
                # If we're at least 80% of the way to the budget, stop
                if tokens_used >= 0.8 * token_budget:
                    break

                # Otherwise, see if we can truncate the item
                words = item.split()
                available_tokens = token_budget - tokens_used
                if available_tokens >= 5:  # Only include if we can get at least 5 words
                    truncated_item = " ".join(words[:available_tokens]) + "..."
                    selected.append(truncated_item)
                break

        return selected

    def _get_token_budget(self, category: str, max_tokens: int) -> int:
        """
        Determine the token budget for each category of information.

        Args:
            category: The category of information
            max_tokens: Total token budget

        Returns:
            Token budget for this category
        """
        # Allocate tokens proportionally to each category
        category_weights = {
            "symptoms": 0.2,  # 20% for symptoms
            "causes": 0.15,  # 15% for causes
            "treatments": 0.4,  # 40% for treatments (most important)
            "prevention": 0.25  # 25% for prevention
        }

        # Calculate the budget
        weight = category_weights.get(category, 0.25)
        return int(max_tokens * weight)

    def _estimate_token_count(self, info: Dict[str, Any]) -> int:
        """
        Estimate the token count for the given information.

        Args:
            info: Dictionary with disease information

        Returns:
            Estimated token count
        """
        # A simple estimation based on word count
        # On average, tokens are ~1.3 words in English
        word_count = 0

        # Count words in plant name
        word_count += len(str(info.get("plant", "")).split())

        # Count words in diseases
        for disease in info.get("diseases", []):
            word_count += len(str(disease.get("name", "")).split())

            for category in ["symptoms", "causes", "treatments", "prevention"]:
                for item in disease.get(category, []):
                    word_count += len(str(item).split())

        # Add overhead for formatting
        word_count += 50

        # Convert to estimated tokens (words * 1.3)
        return int(word_count * 1.3)

    def _reduce_token_count(self, info: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """
        Further reduce token count if needed.

        Args:
            info: Dictionary with disease information
            max_tokens: Maximum tokens allowed

        Returns:
            Reduced information dictionary
        """
        # Create a copy to avoid modifying the original
        reduced_info = {
            "plant": info["plant"],
            "diseases": [],
            "sources": info.get("sources", [])[:1]  # Reduce to just 1 source
        }

        # Only keep the first disease (most relevant one)
        if info.get("diseases"):
            disease = info["diseases"][0]
            reduced_disease = {
                "name": disease["name"],
                "symptoms": [],
                "causes": [],
                "treatments": [],
                "prevention": []
            }

            # Keep only the first few items of each category
            max_items = {
                "symptoms": 2,
                "causes": 1,
                "treatments": 3,
                "prevention": 2
            }

            for category, max_count in max_items.items():
                reduced_disease[category] = disease.get(category, [])[:max_count]

            reduced_info["diseases"].append(reduced_disease)

        return reduced_info


class PlantRecommendationSystem:
    def __init__(self, base_dir="./plant_diseases", lm_studio_url="http://localhost:1234/v1"):
        """
        Initialize the recommendation system with LM Studio and CAG processing.

        Args:
            base_dir: Directory where plant disease information is stored
            lm_studio_url: URL of the LM Studio API server
        """
        self.base_dir = base_dir
        self.lm_studio_url = lm_studio_url
        self.cag_processor = CAGProcessor()

        # Check if base directory exists
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            print(f"Created base directory: {base_dir}")

        # Test LM Studio connection
        try:
            self._test_lm_studio_connection()
            print("✅ Successfully connected to LM Studio")
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to LM Studio at {lm_studio_url}")
            print(f"   Error: {e}")
            print("   Make sure LM Studio is running and the API server is enabled.")
            print("   The system will still work, but will use mock responses instead.")

    def _test_lm_studio_connection(self):
        """Test the connection to LM Studio"""
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

    def load_disease_info(self, plant_name: str, disease_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load disease information from JSON files.

        Args:
            plant_name: Name of the plant
            disease_name: Specific disease to load (optional)

        Returns:
            Dictionary containing disease information
        """
        plant_dir = os.path.join(self.base_dir, plant_name.lower().replace(' ', '_'))

        # Check if plant directory exists
        if not os.path.exists(plant_dir):
            raise FileNotFoundError(f"No information found for plant: {plant_name}. "
                                    f"Run the PlantDiseaseAgent first to gather information.")

        # Determine which JSON file to load
        if disease_name:
            json_path = os.path.join(plant_dir, f"{disease_name.lower().replace(' ', '_')}_info.json")
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

        # Load the JSON data
        with open(json_path, 'r') as f:
            disease_info = json.load(f)

        return disease_info

    def _extract_disease_from_all(self, json_path: str, disease_name: str) -> Dict[str, Any]:
        """
        Extract information for a specific disease from the all_diseases JSON file.

        Args:
            json_path: Path to the all_diseases JSON file
            disease_name: Name of the disease to extract

        Returns:
            Dictionary with information for the specified disease
        """
        with open(json_path, 'r') as f:
            all_info = json.load(f)

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

    def _get_user_details(self) -> Dict[str, str]:
        """
        Get additional details from the user to help with recommendation.

        Returns:
            Dictionary containing user-provided details
        """
        details = {}

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

        return details

    def _generate_recommendation_prompt(self, disease_info: Dict[str, Any], user_details: Dict[str, str],
                                        disease_name: Optional[str] = None, max_tokens: int = 2000) -> str:
        """
        Generate a prompt for the LLM to create personalized recommendations.
        Uses CAG to select the most relevant information based on the user's needs.

        Args:
            disease_info: Dictionary containing disease information
            user_details: Dictionary containing user-provided details
            disease_name: Specific disease to focus on (optional)
            max_tokens: Maximum tokens for context

        Returns:
            Prompt for the LLM
        """
        # Use CAG to extract the most contextually relevant information
        relevant_info = self.cag_processor.extract_relevant_context(
            disease_info, user_details, max_tokens=max_tokens
        )

        plant_name = relevant_info["plant"]

        # Create the base prompt - streamlined version
        prompt = f"""As a plant pathologist, create a personalized recommendation for treating {plant_name} with these disease details:

"""

        # Add disease information from CAG processing
        for disease in relevant_info["diseases"]:
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

        # Add user details - concise format
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
        approx_token_count = len(prompt.split())
        print(f"Approximate token count of prompt: {approx_token_count} words")

        return prompt

    def _call_lm_studio(self, prompt: str) -> str:
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
                    {"role": "system",
                     "content": "You are a plant disease expert providing practical treatment recommendations."},
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
        More sophisticated than a simple mock response.

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
        organic_treatments = [
            "Apply neem oil (2 tbsp per gallon of water) to all affected parts weekly for 3 weeks.",
            "Use a copper-based organic fungicide labeled for fruit trees.",
            "Apply compost tea as a foliar spray to boost plant immunity.",
            "Spray with a solution of 1 tablespoon baking soda, 1 teaspoon mild soap, and 1 gallon of water."
        ]

        conventional_treatments = [
            "Apply a copper-based fungicide according to package directions.",
            "Use a systemic fungicide containing chlorothalonil for more severe cases.",
            "Apply a broad-spectrum fungicide that targets this specific pathogen.",
            "Treat with appropriate bactericide if bacterial infection is confirmed."
        ]

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

    def generate_recommendation(self, plant_name: str, disease_name: Optional[str] = None,
                                output_file: Optional[str] = None) -> str:
        """
        Generate a personalized recommendation for treating a plant disease.

        Args:
            plant_name: Name of the plant
            disease_name: Specific disease to focus on (optional)
            output_file: File to save the recommendation to (optional)

        Returns:
            The generated recommendation
        """
        try:
            # Load disease information
            disease_info = self.load_disease_info(plant_name, disease_name)

            # Get additional details from the user
            print("\nTo provide a personalized recommendation, please answer a few questions:")
            user_details = self._get_user_details()

            # Determine the model's token limit
            model_token_limit = self._get_model_token_limit()

            # Generate prompt for the LLM
            prompt = self._generate_recommendation_prompt(
                disease_info,
                user_details,
                disease_name,
                max_tokens=model_token_limit - 1000  # Leave room for the response
            )

            print("\nGenerating personalized recommendation using LM Studio...")

            # Call the LM Studio API
            recommendation = self._call_lm_studio(prompt)

            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(recommendation)
                print(f"Recommendation saved to {output_file}")

            return recommendation

        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def _get_model_token_limit(self) -> int:
        """
        Try to determine the token limit of the loaded model in LM Studio.
        Falls back to a conservative estimate if not determinable.

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
                    if "70b" in model_id:
                        return 8192  # 70B models often support 8K context
                    elif "13b" in model_id or "15b" in model_id:
                        return 4096  # 13B/15B models typically support 4K
                    elif "8b" in model_id or "7b" in model_id:
                        return 4096  # 8B/7B models typically support 4K
                    elif "3.5" in model_id and "sonnet" in model_id:
                        return 8192  # Claude 3.5 Sonnet-like support 8K
                    elif "3.5" in model_id and "haiku" in model_id:
                        return 4096  # Claude 3.5 Haiku-like support 4K
                    elif "3" in model_id and "opus" in model_id:
                        return 16384  # Claude 3 Opus-like support 16K

            # Conservative default
            return 2048
        except:
            # Very conservative fallback
            return 2048


def install_requirements():
    """Install required packages if they're not already installed."""
    try:
        import sklearn
        print("sklearn already installed.")
    except ImportError:
        print("Installing scikit-learn...")
        import subprocess
        subprocess.check_call(["pip", "install", "scikit-learn"])
        print("scikit-learn installed successfully.")


def main():
    # Check and install requirements
    install_requirements()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate personalized plant disease treatment recommendations using CAG and LM Studio')
    parser.add_argument('plant', help='Name of the plant')
    parser.add_argument('--disease', help='Specific disease name (optional)')
    parser.add_argument('--data-dir', help='Directory with plant disease data (default: ./plant_diseases)',
                        default='./plant_diseases')
    parser.add_argument('--output', help='Output file for the recommendation (optional)')
    parser.add_argument('--lm-studio-url', help='URL for LM Studio API (default: http://localhost:1234/v1)',
                        default='http://localhost:1234/v1')

    args = parser.parse_args()

    # Create recommendation system
    recommender = PlantRecommendationSystem(
        base_dir=args.data_dir,
        lm_studio_url=args.lm_studio_url
    )

    # Generate recommendation
    recommendation = recommender.generate_recommendation(
        plant_name=args.plant,
        disease_name=args.disease,
        output_file=args.output
    )

    # Print the recommendation
    print("\n" + "=" * 80)
    print("PERSONALIZED TREATMENT RECOMMENDATION")
    print("=" * 80 + "\n")
    print(recommendation)


# Alternative usage as a web interface
def create_flask_app():
    from flask import Flask, request, jsonify, render_template

    app = Flask(__name__)
    recommender = PlantRecommendationSystem()

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/recommend', methods=['POST'])
    def recommend():
        data = request.json
        plant_name = data.get('plant')
        disease_name = data.get('disease')
        user_details = data.get('details', {})

        try:
            # Load disease information
            disease_info = recommender.load_disease_info(plant_name, disease_name)

            # Generate prompt for the LLM
            model_token_limit = recommender._get_model_token_limit()
            prompt = recommender._generate_recommendation_prompt(
                disease_info,
                user_details,
                disease_name,
                max_tokens=model_token_limit - 1000
            )

            # Call the LM Studio API
            recommendation = recommender._call_lm_studio(prompt)

            return jsonify({'recommendation': recommendation})

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return app


if __name__ == "__main__":
    main()