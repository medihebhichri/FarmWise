import os
import json
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# ===========================
# Configuration Section
# ===========================
# LM Studio configuration
LM_SERVER_URL = "http://localhost:1234"  # Update if necessary
COMPLETIONS_ENDPOINT = f"{LM_SERVER_URL}/v1/completions"
DEFAULT_MODEL = "your-model"  # e.g., "llama-2-7b"

# RAG configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default sentence embedding model from sentence-transformers
TOP_K_RESULTS = 5  # Number of most relevant results to retrieve for each query

# Define the fixed keys for growth factors - IMPORTANT: Moving this outside the class
GROWTH_FACTOR_KEYS = [
    "temperature",
    "humidity",
    "method_use",
    "soil_natural",
    "method_use_for_plant",
    "light_exposure",
    "water_requirements",
    "fertilizer",
    "pH_level",
    "plant_spacing"
]


class PlantRAGRecommender:
    def __init__(self, base_directory="."):
        """
        Initialize the RAG system for plant growth recommendations.

        Args:
            base_directory: Directory containing plant folders with summaries
        """
        self.base_directory = base_directory
        self.plant_data = {}
        self.embeddings = {}
        self.embedding_model = None
        self.embedding_loaded = False

        # Try to load the embedding model
        try:
            print("Loading sentence embedding model...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.embedding_loaded = True
            print(f"Successfully loaded {EMBEDDING_MODEL}")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            print("Will continue without semantic search capabilities.")

    def load_all_plant_data(self):
        """
        Load all plant data from the summaries folders.
        Returns the number of plants loaded.
        """
        # Reset plant data
        self.plant_data = {}
        self.embeddings = {}

        # Search for all summary JSON files
        pattern = os.path.join(self.base_directory, "*", "summaries", "*.json")
        summary_files = glob.glob(pattern)

        print(f"Found {len(summary_files)} plant summary files")

        for file_path in summary_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                plant_name = data.get("plant_name", os.path.basename(file_path).split("_")[0])
                self.plant_data[plant_name.lower()] = data

                # Create content for embedding
                if self.embedding_loaded:
                    content = self._prepare_content_for_embedding(data)
                    embedding = self.embedding_model.encode(content)
                    self.embeddings[plant_name.lower()] = {
                        "content": content,
                        "embedding": embedding
                    }

                print(f"Loaded data for {plant_name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return len(self.plant_data)

    def _prepare_content_for_embedding(self, plant_data):
        """
        Prepare plant data for embedding by extracting relevant text.
        """
        content = f"Plant: {plant_data.get('plant_name', 'Unknown')}\n"

        # Extract summaries and growth factors from videos
        for video in plant_data.get("videos", []):
            content += f"Summary: {video.get('summary', '')}\n"

            growth_factors = video.get("growth_factors", {})
            content += "Growth Factors:\n"
            for factor, value in growth_factors.items():
                content += f"- {factor}: {value}\n"

        return content

    def find_similar_plants(self, query_plant, top_k=3):
        """
        Find plants similar to the query plant based on embedding similarity.

        Args:
            query_plant: Name of the plant to find similarities for
            top_k: Number of similar plants to return

        Returns:
            List of similar plant names or empty list if not found
        """
        if not self.embedding_loaded:
            print("Embedding model not loaded, cannot find similar plants")
            return []

        query_plant = query_plant.lower()

        # If we don't have the exact plant, encode the query
        if query_plant not in self.embeddings:
            query_embedding = self.embedding_model.encode(f"Plant: {query_plant}")
        else:
            query_embedding = self.embeddings[query_plant]["embedding"]

        # Calculate similarities
        similarities = {}
        for plant_name, plant_data in self.embeddings.items():
            if plant_name == query_plant:
                continue

            similarity = np.dot(query_embedding, plant_data["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(plant_data["embedding"])
            )
            similarities[plant_name] = similarity

        # Sort by similarity
        sorted_plants = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Return top_k similar plants
        return [plant for plant, _ in sorted_plants[:top_k]]

    def get_optimal_growth_factors(self, plant_name):
        """
        Get the optimal growth factors for a specific plant.
        If plant is not in database, try to find similar plants.

        Args:
            plant_name: Name of the plant

        Returns:
            Dictionary of growth factors or None if not found
        """
        plant_name_lower = plant_name.lower()

        # Direct match
        if plant_name_lower in self.plant_data:
            return self._extract_optimal_factors(self.plant_data[plant_name_lower])

        # Try to find similar plants
        similar_plants = self.find_similar_plants(plant_name_lower)

        if not similar_plants:
            return None

        # Combine growth factors from similar plants
        combined_factors = {}
        for similar_plant in similar_plants:
            if similar_plant in self.plant_data:
                plant_factors = self._extract_optimal_factors(self.plant_data[similar_plant])

                for factor, value in plant_factors.items():
                    if factor not in combined_factors:
                        combined_factors[factor] = []

                    if value != "Not specified":
                        combined_factors[factor].append(value)

        # Consolidate the combined factors
        optimal_factors = {}
        for factor, values in combined_factors.items():
            if values:
                # For now, just take the most common value
                optimal_factors[factor] = max(set(values), key=values.count)
            else:
                optimal_factors[factor] = "Not specified"

        return optimal_factors

    def _extract_optimal_factors(self, plant_data):
        """
        Extract the optimal growth factors from a plant's data.
        This uses a simple approach of taking the most common values across videos.

        Args:
            plant_data: JSON data for a plant

        Returns:
            Dictionary of optimal growth factors
        """
        # Use the global GROWTH_FACTOR_KEYS variable
        factor_values = {factor: [] for factor in GROWTH_FACTOR_KEYS}

        # Collect all values for each factor
        for video in plant_data.get("videos", []):
            growth_factors = video.get("growth_factors", {})

            for factor in GROWTH_FACTOR_KEYS:
                value = growth_factors.get(factor, "Not specified")
                if value != "Not specified":
                    factor_values[factor].append(value)

        # Determine the optimal value for each factor
        optimal_factors = {}
        for factor, values in factor_values.items():
            if values:
                # Take the most common value as optimal
                optimal_factors[factor] = max(set(values), key=values.count)
            else:
                optimal_factors[factor] = "Not specified"

        return optimal_factors

    def generate_recommendation(self, plant_name):
        """
        Generate a comprehensive recommendation for growing the plant.

        Args:
            plant_name: Name of the plant

        Returns:
            Generated recommendation text
        """
        plant_name_lower = plant_name.lower()

        # Check if we have the plant in our database
        direct_match = plant_name_lower in self.plant_data

        # Get optimal factors
        optimal_factors = self.get_optimal_growth_factors(plant_name_lower)

        if not optimal_factors:
            return f"Sorry, I don't have enough information about {plant_name} to make recommendations."

        # Find similar plants if this is not a direct match
        similar_plants = []
        if not direct_match:
            similar_plants = self.find_similar_plants(plant_name_lower)

        # Create context for LLM
        context = f"Plant name: {plant_name}\n\n"
        context += "Optimal growth factors:\n"

        for factor, value in optimal_factors.items():
            context += f"- {factor}: {value}\n"

        if similar_plants:
            context += f"\nNote: These recommendations are based on similar plants: {', '.join(similar_plants)}\n"

        # Generate recommendation using LLM
        prompt = (
            f"Based on the following information about {plant_name}, create a detailed and helpful recommendation "
            f"for growing this plant successfully. Include specific advice for each factor when available.\n\n"
            f"{context}\n\n"
            f"Write a comprehensive guide that explains the optimal growing conditions and practices for {plant_name}. "
            f"Make the recommendations practical and easy to follow."
        )

        return self._generate_with_lm(prompt)

    def _generate_with_lm(self, prompt):
        """
        Generate text using LM Studio.

        Args:
            prompt: Prompt for the language model

        Returns:
            Generated text or error message
        """
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.7
        }

        try:
            response = requests.post(COMPLETIONS_ENDPOINT, json=payload)
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0].get("text", "").strip()
                else:
                    return "Generation failed: No choices in response."
            else:
                return f"Error: LM Studio responded with status {response.status_code}"
        except Exception as e:
            return f"Exception during generation: {e}"


def main():
    print("Plant Growth Factors RAG Recommendation System")
    print("==============================================")

    # Initialize the RAG system
    recommender = PlantRAGRecommender()

    # Load all plant data
    num_plants = recommender.load_all_plant_data()
    if num_plants == 0:
        print("No plant data found. Please run the data collection script first.")
        return

    print(f"Loaded data for {num_plants} plants")

    while True:
        print("\nOptions:")
        print("1. Get recommendations for a plant")
        print("2. Find similar plants")
        print("3. List all available plants")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            plant_name = input("Enter the plant name: ").strip()
            if not plant_name:
                continue

            print(f"\nGenerating recommendations for {plant_name}...")
            recommendation = recommender.generate_recommendation(plant_name)

            print("\n" + "=" * 50)
            print(f"RECOMMENDATIONS FOR {plant_name.upper()}")
            print("=" * 50)
            print(recommendation)
            print("=" * 50)

        elif choice == "2":
            plant_name = input("Enter the plant name: ").strip()
            if not plant_name:
                continue

            similar_plants = recommender.find_similar_plants(plant_name)

            print(f"\nPlants similar to {plant_name}:")
            if similar_plants:
                for i, plant in enumerate(similar_plants, 1):
                    print(f"{i}. {plant}")
            else:
                print("No similar plants found or embedding model not available.")

        elif choice == "3":
            print("\nAvailable plants:")
            for i, plant in enumerate(sorted(recommender.plant_data.keys()), 1):
                print(f"{i}. {plant}")

        elif choice == "4":
            print("Thank you for using the Plant Growth Factors Recommender!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()