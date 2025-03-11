"""
Context analyzer for selecting the most relevant information based on the user's situation.
Performs contextual analysis to extract the most relevant information from the disease data.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import re

from config import CATEGORY_WEIGHTS
import utils


class ContextAnalyzer:
    def __init__(self):
        """Initialize the context analyzer."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000
            )
            self.sklearn_available = True
            print("Using scikit-learn for advanced text matching")
        except ImportError:
            print("Warning: scikit-learn not available. Using basic text matching instead.")
            self.sklearn_available = False

    def extract_relevant_context(self, disease_info: Dict[str, Any], user_query: Dict[str, str],
                                 max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Extract the most relevant information based on the user's situation and query.

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
                selected_items = utils.select_items_by_token_budget(items, scores, token_budget)

                processed_disease[category] = selected_items

            relevant_info["diseases"].append(processed_disease)

        # Further filter if we still have too many tokens
        estimated_tokens = utils.estimate_json_token_count(relevant_info)
        if estimated_tokens > max_tokens:
            # Apply more aggressive filtering
            return self._reduce_token_count(relevant_info, max_tokens)

        return relevant_info

    def _create_user_context(self, user_query: Dict[str, str]) -> str:
        """
        Create a consolidated context string from user information.

        Args:
            user_query: Dictionary with user details

        Returns:
            String representing the user's context
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
        Calculate relevance scores using TF-IDF and cosine similarity.

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
            # Fallback if vectorization fails
            return np.ones(len(items))

    def _calculate_relevance_basic(self, items: List[str], context: str) -> List[float]:
        """
        Calculate relevance scores using basic text matching.

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

    def _get_token_budget(self, category: str, max_tokens: int) -> int:
        """
        Determine the token budget for each category of information.

        Args:
            category: The category of information
            max_tokens: Total token budget

        Returns:
            Token budget for this category
        """
        # Get the weight for this category
        weight = CATEGORY_WEIGHTS.get(category, 0.25)

        # Calculate and return the budget
        return int(max_tokens * weight)

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


# Simple test function
def test_analyzer():
    from data_processor import DataProcessor

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

        print(f"Filtered to relevant information for user context:")
        for disease in relevant["diseases"]:
            print(f"- {disease['name']}")
            print(f"  Symptoms: {len(disease['symptoms'])} items")
            print(f"  Treatments: {len(disease['treatments'])} items")

    except FileNotFoundError:
        print("No existing data found. Run the data processor first.")


if __name__ == "__main__":
    test_analyzer()