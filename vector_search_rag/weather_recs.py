import pymongo
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Connect to MongoDB
client = pymongo.MongoClient(
    "mongodb+srv://hichriiheb13:3eNgGXS0RnPiahuD@cluster0.kxf36.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.sample_weatherdata
collection = db.data

# Hugging Face API settings
hf_token = ""
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def generate_embedding(text: str) -> list:
    """Generate embedding for given text using Hugging Face API."""
    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text}
    )
    if response.status_code == 200:
        embedding = response.json()
        if isinstance(embedding, list) and all(isinstance(x, float) for x in embedding):
            return embedding
        else:
            print("Unexpected embedding format:", embedding)
            return []
    else:
        print("Error generating embedding:", response.text)
        return []


def precompute_embeddings():
    """Precompute and store coordinate embeddings for all documents."""
    for doc in collection.find():
        # Skip if embedding already exists
        if "coordinate_embedding" in doc:
            continue

        # Extract coordinates from the document
        position = doc.get("position", {})
        coordinates = position.get("coordinates", [])
        if len(coordinates) < 2:
            print(f"Skipping document {doc['_id']} due to missing coordinates.")
            continue

        lon = coordinates[0]
        lat = coordinates[1]
        embedding_text = f"Coordinates: {lon},{lat}"
        embedding = generate_embedding(embedding_text)

        if embedding:
            # Update the document with the embedding
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"coordinate_embedding": embedding}}
            )
            print(f"Updated document {doc['_id']} with embedding.")
        else:
            print(f"Failed to generate embedding for document {doc['_id']}.")


def find_nearest_places(query_coordinates, top_n=5):
    """Find the nearest places based on coordinate embeddings."""
    # Generate embedding for the query coordinates
    query_embedding = generate_embedding(
        f"Coordinates: {query_coordinates[0]},{query_coordinates[1]}"
    )
    if not query_embedding:
        print("Query embedding could not be generated.")
        return []

    # Collect all documents with precomputed embeddings
    embeddings = []
    doc_ids = []
    for doc in collection.find({"coordinate_embedding": {"$exists": True}}):
        doc_embedding = doc["coordinate_embedding"]
        if len(doc_embedding) != len(query_embedding):
            print(f"Skipping document {doc['_id']} due to incompatible embedding length.")
            continue
        embeddings.append(doc_embedding)
        doc_ids.append(doc["_id"])

    if not embeddings:
        print("No valid embeddings found in the database.")
        return []


    similarities = cosine_similarity([query_embedding], embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]


    top_results = []
    for idx in top_indices:
        doc = collection.find_one({"_id": doc_ids[idx]})
        top_results.append({
            "document_id": doc_ids[idx],
            "similarity": similarities[0][idx],
            "document": doc
        })

    return top_results

if __name__ == "__main__":
    query_coords = [47.600, 47.900]  # Example coordinates [longitude, latitude]
    results = find_nearest_places(query_coords, top_n=5)

    if results:
        for result in results:
            print(f"Document ID: {result['document_id']}")
            print(f"Similarity Score: {result['similarity']:.4f}")
            pos = result['document'].get('position', {})
            coords = pos.get('coordinates', [])
            print(f"Coordinates: {coords[0]}, {coords[1]}" if len(coords) >= 2 else "No coordinates")
            print("------")
    else:
        print("No results found.")