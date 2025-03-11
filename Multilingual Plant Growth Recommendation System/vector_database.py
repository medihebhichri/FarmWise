"""
vector_database.py
Manages the vector database for plant information storage and retrieval.
Provides semantic search capabilities across multiple languages.
"""

import os
import json
import pickle
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple

# Import configuration
from config import (
    DATA_DIR, VECTOR_DB_DIR, LOGS_DIR, EMBEDDING_MODEL,
    EMBEDDING_DIMENSION, GROWTH_FACTOR_KEYS
)

# Set up logging
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "vector_database.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VectorDatabase")


class VectorDatabase:
    """Manages vector embeddings for plant data with multi-language support."""

    def __init__(self):
        """Initialize the vector database."""
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        self.embedding_dim = EMBEDDING_DIMENSION

        # Ensure vector database directory exists
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)

        # Initialize embedding model
        self._load_embedding_model()

        # Load or create database
        self._load_or_create_database()

    def _load_embedding_model(self):
        """Load the multilingual sentence embedding model."""
        try:
            logger.info(f"Loading sentence embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Successfully loaded model with dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            logger.warning("Vector database will not be functional without embedding model")

    def _create_index(self):
        """Create a new FAISS index for vector storage."""
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")
            return True
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return False

    def _load_or_create_database(self):
        """Load existing database or create a new one if not found."""
        index_path = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
        metadata_path = os.path.join(VECTOR_DB_DIR, "metadata.pkl")
        documents_path = os.path.join(VECTOR_DB_DIR, "documents.pkl")

        # Check if database files exist
        if (os.path.exists(index_path) and
                os.path.exists(metadata_path) and
                os.path.exists(documents_path)):
            try:
                # Load index
                self.index = faiss.read_index(index_path)

                # Load metadata and documents
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)

                with open(documents_path, 'rb') as f:
                    self.documents = pickle.load(f)

                logger.info(f"Loaded existing database with {self.index.ntotal} vectors")
                return True
            except Exception as e:
                logger.error(f"Error loading existing database: {e}")
                logger.info("Creating new database")

        # Create new database
        success = self._create_index()
        if success:
            self.metadata = []
            self.documents = []
            return True
        else:
            return False

    def _save_database(self):
        """Save the database to disk."""
        if self.index is None:
            logger.error("Cannot save database: index is not initialized")
            return False

        try:
            index_path = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
            metadata_path = os.path.join(VECTOR_DB_DIR, "metadata.pkl")
            documents_path = os.path.join(VECTOR_DB_DIR, "documents.pkl")

            # Save index
            faiss.write_index(self.index, index_path)

            # Save metadata and documents
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

            with open(documents_path, 'wb') as f:
                pickle.dump(self.documents, f)

            logger.info(f"Saved database with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            return False

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Create embedding vector for text.

        Args:
            text (str): Text to embed

        Returns:
            numpy.ndarray: Embedding vector or None if failed
        """
        if self.embedding_model is None:
            logger.error("Cannot embed text: embedding model not loaded")
            return None

        try:
            embedding = self.embedding_model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return None

    def _prepare_document_chunks(self, plant_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Break down plant data into manageable chunks for embedding.

        Args:
            plant_data (dict): Plant data collected from various sources

        Returns:
            list: List of document chunks with metadata
        """
        chunks = []
        plant_name = plant_data.get("plant_name", "Unknown plant")
        language = plant_data.get("language", "en")
        dialect = plant_data.get("dialect")

        # 1. Process YouTube data
        for i, video in enumerate(plant_data.get("youtube_data", [])):
            # Add video summary
            if video.get("summary"):
                chunks.append({
                    "text": f"Plant: {plant_name}\nSource: YouTube video\n{video.get('summary', '')}",
                    "metadata": {
                        "plant_name": plant_name,
                        "language": language,
                        "dialect": dialect,
                        "source_type": "youtube",
                        "source_id": video.get("video_id", f"video_{i}"),
                        "chunk_type": "summary"
                    }
                })

            # Add growth factors as structured data
            growth_factors = video.get("growth_factors", {})
            if growth_factors:
                factor_text = f"Plant: {plant_name}\nSource: YouTube video\n"
                for factor, value in growth_factors.items():
                    if value and value != "Not specified":
                        factor_text += f"{factor}: {value}\n"

                chunks.append({
                    "text": factor_text,
                    "metadata": {
                        "plant_name": plant_name,
                        "language": language,
                        "dialect": dialect,
                        "source_type": "youtube",
                        "source_id": video.get("video_id", f"video_{i}"),
                        "chunk_type": "growth_factors"
                    }
                })

        # 2. Process web data
        for i, website in enumerate(plant_data.get("web_data", [])):
            # Add website summary
            if website.get("summary"):
                chunks.append({
                    "text": f"Plant: {plant_name}\nSource: Gardening website\n{website.get('summary', '')}",
                    "metadata": {
                        "plant_name": plant_name,
                        "language": language,
                        "dialect": dialect,
                        "source_type": "website",
                        "source_id": f"website_{i}",
                        "source_url": website.get("url", ""),
                        "chunk_type": "summary"
                    }
                })

            # Add growth factors
            growth_factors = website.get("growth_factors", {})
            if growth_factors:
                factor_text = f"Plant: {plant_name}\nSource: Gardening website\n"
                for factor, value in growth_factors.items():
                    if value and value != "Not specified":
                        factor_text += f"{factor}: {value}\n"

                chunks.append({
                    "text": factor_text,
                    "metadata": {
                        "plant_name": plant_name,
                        "language": language,
                        "dialect": dialect,
                        "source_type": "website",
                        "source_id": f"website_{i}",
                        "source_url": website.get("url", ""),
                        "chunk_type": "growth_factors"
                    }
                })

        # 3. Process Wikipedia data
        wiki_data = plant_data.get("wikipedia_data", {})
        if wiki_data:
            # Add Wikipedia summary
            if wiki_data.get("summary"):
                chunks.append({
                    "text": f"Plant: {plant_name}\nSource: Wikipedia\n{wiki_data.get('summary', '')}",
                    "metadata": {
                        "plant_name": plant_name,
                        "language": language,
                        "dialect": dialect,
                        "source_type": "wikipedia",
                        "source_id": "wiki_summary",
                        "source_url": wiki_data.get("url", ""),
                        "chunk_type": "summary"
                    }
                })

            # Add growth factors
            growth_factors = wiki_data.get("growth_factors", {})
            if growth_factors:
                factor_text = f"Plant: {plant_name}\nSource: Wikipedia\n"
                for factor, value in growth_factors.items():
                    if value and value != "Not specified":
                        factor_text += f"{factor}: {value}\n"

                chunks.append({
                    "text": factor_text,
                    "metadata": {
                        "plant_name": plant_name,
                        "language": language,
                        "dialect": dialect,
                        "source_type": "wikipedia",
                        "source_id": "wiki_factors",
                        "source_url": wiki_data.get("url", ""),
                        "chunk_type": "growth_factors"
                    }
                })

            # Add content chunks if available
            content = wiki_data.get("content", "")
            if content and len(content) > 100:
                # Split content into paragraphs
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

                for i, para in enumerate(paragraphs):
                    if len(para) < 50:  # Skip very short paragraphs
                        continue

                    chunks.append({
                        "text": f"Plant: {plant_name}\nSource: Wikipedia\n{para}",
                        "metadata": {
                            "plant_name": plant_name,
                            "language": language,
                            "dialect": dialect,
                            "source_type": "wikipedia",
                            "source_id": f"wiki_para_{i}",
                            "source_url": wiki_data.get("url", ""),
                            "chunk_type": "content"
                        }
                    })

        return chunks

    def index_plant_data(self, plant_data: Dict[str, Any]) -> bool:
        """
        Index plant data into the vector database.

        Args:
            plant_data (dict): Plant data to index

        Returns:
            bool: True if successful, False otherwise
        """
        if self.embedding_model is None or self.index is None:
            logger.error("Cannot index data: model or index not initialized")
            return False

        plant_name = plant_data.get("plant_name", "Unknown plant")
        language = plant_data.get("language", "en")

        try:
            # Prepare document chunks
            chunks = self._prepare_document_chunks(plant_data)

            if not chunks:
                logger.warning(f"No chunks created for {plant_name} in {language}")
                return False

            logger.info(f"Created {len(chunks)} chunks for {plant_name} in {language}")

            # Embed and add each chunk
            vectors = []
            new_documents = []
            new_metadata = []

            for chunk in chunks:
                text = chunk["text"]
                metadata = chunk["metadata"]

                # Create embedding
                embedding = self._embed_text(text)

                if embedding is not None:
                    vectors.append(embedding)
                    new_documents.append(text)
                    new_metadata.append(metadata)

            # Add to index
            if vectors:
                vectors_array = np.array(vectors).astype('float32')
                self.index.add(vectors_array)

                # Update documents and metadata
                self.documents.extend(new_documents)
                self.metadata.extend(new_metadata)

                # Save database
                self._save_database()

                logger.info(f"Indexed {len(vectors)} vectors for {plant_name} in {language}")
                return True
            else:
                logger.warning(f"No vectors created for {plant_name} in {language}")
                return False

        except Exception as e:
            logger.error(f"Error indexing plant data: {e}")
            return False

    def search(self, query: str, language_code: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector database for similar content.

        Args:
            query (str): Search query
            language_code (str, optional): Filter results by language
            top_k (int): Number of results to return

        Returns:
            list: List of search results with metadata
        """
        if self.embedding_model is None or self.index is None:
            logger.error("Cannot search: model or index not initialized")
            return []

        if self.index.ntotal == 0:
            logger.warning("Database is empty, no results to return")
            return []

        try:
            # Create query embedding
            query_embedding = self._embed_text(query)

            if query_embedding is None:
                logger.error("Failed to create query embedding")
                return []

            # Search index
            query_embedding = np.array([query_embedding]).astype('float32')
            distances, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))

            # Process results
            results = []
            for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= len(self.documents):
                    continue

                text = self.documents[idx]
                metadata = self.metadata[idx]

                # Filter by language if specified
                if language_code and metadata.get("language") != language_code:
                    continue

                results.append({
                    "text": text,
                    "metadata": metadata,
                    "distance": float(distance),
                    "rank": rank
                })

                if len(results) >= top_k:
                    break

            logger.info(f"Search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching database: {e}")
            return []

    def search_by_plant(self, plant_name: str, language_code: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for information about a specific plant.

        Args:
            plant_name (str): Name of the plant
            language_code (str, optional): Filter results by language
            top_k (int): Number of results to return

        Returns:
            list: List of search results with metadata
        """
        # Create a plant-specific query
        query = f"Plant: {plant_name}"
        results = self.search(query, language_code, top_k)

        # Filter results to ensure they are for the requested plant
        plant_results = []
        for result in results:
            result_plant = result["metadata"].get("plant_name", "").lower()
            if result_plant and plant_name.lower() in result_plant:
                plant_results.append(result)

        return plant_results

    def get_growth_factors(self, plant_name: str, language_code: str = None) -> Dict[str, Any]:
        """
        Get consolidated growth factors for a specific plant.

        Args:
            plant_name (str): Name of the plant
            language_code (str, optional): Filter by language

        Returns:
            dict: Consolidated growth factors
        """
        # Search specifically for growth factor chunks
        query = f"Plant: {plant_name} growth factors"
        results = self.search(query, language_code, top_k=10)

        # Filter for growth factor chunks about this plant
        factor_results = []
        for result in results:
            metadata = result["metadata"]
            result_plant = metadata.get("plant_name", "").lower()
            chunk_type = metadata.get("chunk_type", "")

            if result_plant and plant_name.lower() in result_plant and chunk_type == "growth_factors":
                factor_results.append(result)

        if not factor_results:
            logger.warning(f"No growth factor information found for {plant_name}")
            return {factor: "Not specified" for factor in GROWTH_FACTOR_KEYS}

        # Extract and consolidate growth factors
        factors_data = {}
        for result in factor_results:
            text = result["text"]

            # Extract factors from text
            for factor in GROWTH_FACTOR_KEYS:
                if factor in text:
                    lines = text.split('\n')
                    for line in lines:
                        if line.startswith(f"{factor}:"):
                            value = line.split(':', 1)[1].strip()

                            if factor not in factors_data:
                                factors_data[factor] = []

                            if value and value != "Not specified":
                                factors_data[factor].append(value)

        # Consolidate factors (take most frequent value)
        consolidated_factors = {}
        for factor in GROWTH_FACTOR_KEYS:
            values = factors_data.get(factor, [])
            if values:
                # Get most frequent value
                value_count = {}
                for value in values:
                    value_count[value] = value_count.get(value, 0) + 1

                most_frequent = max(value_count.items(), key=lambda x: x[1])
                consolidated_factors[factor] = most_frequent[0]
            else:
                consolidated_factors[factor] = "Not specified"

        return consolidated_factors

    def find_similar_plants(self, plant_name: str, language_code: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find plants similar to the query plant.

        Args:
            plant_name (str): Name of the plant
            language_code (str, optional): Filter by language
            top_k (int): Number of similar plants to return

        Returns:
            list: List of similar plants with similarity scores
        """
        # Get plant information
        plant_data = self.search_by_plant(plant_name, language_code, 5)

        if not plant_data:
            logger.warning(f"No information found for {plant_name}")
            return []

        # Create a comprehensive query combining information about the plant
        query_texts = []
        for item in plant_data:
            if "growth_factors" in item["text"]:
                query_texts.append(item["text"])

        if not query_texts:
            # Use any available information if no growth factors found
            query_texts = [item["text"] for item in plant_data[:2]]

        combined_query = "\n".join(query_texts)

        # Search for similar plant information
        results = self.search(combined_query, language_code, top_k * 3)

        # Extract unique plants from results
        similar_plants = {}
        for result in results:
            result_plant = result["metadata"].get("plant_name", "").lower()

            # Skip if it's the same plant
            if result_plant and plant_name.lower() in result_plant:
                continue

            # Add to similar plants if not already present
            if result_plant and result_plant not in similar_plants:
                similar_plants[result_plant] = {
                    "plant_name": result["metadata"].get("plant_name"),
                    "similarity_score": 1.0 / (1.0 + result["distance"]),  # Convert distance to similarity
                    "language": result["metadata"].get("language"),
                    "sample_text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
                }

        # Sort by similarity score and return top k
        similar_plant_list = sorted(similar_plants.values(), key=lambda x: x["similarity_score"], reverse=True)
        return similar_plant_list[:top_k]

    def get_all_indexed_plants(self, language_code: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of all plants in the database.

        Args:
            language_code (str, optional): Filter by language

        Returns:
            dict: Dictionary of plants with metadata
        """
        plants = {}

        for metadata in self.metadata:
            plant_name = metadata.get("plant_name")
            lang = metadata.get("language")

            # Skip if language doesn't match filter
            if language_code and lang != language_code:
                continue

            if plant_name and plant_name not in plants:
                plants[plant_name] = {
                    "name": plant_name,
                    "language": lang,
                    "dialect": metadata.get("dialect")
                }

        return plants

    def clear_database(self) -> bool:
        """
        Clear the entire database.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create new index
            self._create_index()

            # Reset documents and metadata
            self.documents = []
            self.metadata = []

            # Save empty database
            self._save_database()

            logger.info("Database cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

    def index_plant_from_file(self, file_path: str) -> bool:
        """
        Index plant data from a JSON file.

        Args:
            file_path (str): Path to JSON file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                plant_data = json.load(f)

            return self.index_plant_data(plant_data)
        except Exception as e:
            logger.error(f"Error loading plant data from {file_path}: {e}")
            return False

    def index_all_plant_files(self, directory: str = DATA_DIR) -> int:
        """
        Index all plant data files in a directory and its subdirectories.

        Args:
            directory (str): Directory to search for JSON files

        Returns:
            int: Number of files successfully indexed
        """
        indexed_count = 0

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith("_data.json"):
                    file_path = os.path.join(root, file)

                    logger.info(f"Indexing file: {file_path}")
                    success = self.index_plant_from_file(file_path)

                    if success:
                        indexed_count += 1

        logger.info(f"Indexed {indexed_count} plant data files")
        return indexed_count

# For testing
if __name__ == "__main__":
    vector_db = VectorDatabase()

    # Test indexing all available plant data
    print(f"Starting database with {vector_db.index.ntotal if vector_db.index else 0} vectors")

    choice = input("Index all plant data? (y/n): ").strip().lower()
    if choice == 'y':
        indexed = vector_db.index_all_plant_files()
        print(f"Indexed {indexed} files. Database now has {vector_db.index.ntotal} vectors")

    # Test search
    while True:
        query = input("\nEnter search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        language = input("Enter language code (press Enter for any): ").strip() or None
        results = vector_db.search(query, language_code=language, top_k=3)

        print(f"\nResults for '{query}':")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Plant: {result['metadata'].get('plant_name')}")
            print(f"Source: {result['metadata'].get('source_type')}")
            print(f"Language: {result['metadata'].get('language')}")
            print(f"Text: {result['text'][:150]}..." if len(result['text']) > 150 else result['text'])