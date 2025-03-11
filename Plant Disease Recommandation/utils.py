import os
import re
import json
import hashlib
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional, Union


def create_directory_if_not_exists(directory_path: str) -> None:
    """Create a directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)


def get_plant_directory(base_dir: str, plant_name: str) -> str:
    """Get the directory path for a specific plant."""
    return os.path.join(base_dir, plant_name.lower().replace(' ', '_'))


def get_cache_directory(base_dir: str, plant_name: str) -> str:
    """Get the cache directory for a specific plant."""
    plant_dir = get_plant_directory(base_dir, plant_name)
    cache_dir = os.path.join(plant_dir, "cache")
    create_directory_if_not_exists(cache_dir)
    return cache_dir


def clean_text(text: str) -> str:
    """Clean and format extracted text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove javascript snippets
    text = re.sub(r'var\s+\w+\s*=\s*.*?;', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    return text


def is_domain_trusted(url: str, trusted_domains: List[str]) -> bool:
    """Check if a URL's domain is in the trusted domains list."""
    domain = urlparse(url).netloc
    return any(trusted in domain for trusted in trusted_domains)


def get_cache_filename(cache_dir: str, url: str) -> str:
    """Generate a cache filename for a URL."""
    domain = urlparse(url).netloc
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(cache_dir, f"{domain}_{url_hash}.json")


def load_from_cache(cache_filename: str) -> Optional[Dict[str, Any]]:
    """Load data from cache if it exists."""
    if os.path.exists(cache_filename):
        with open(cache_filename, 'r') as f:
            return json.load(f)
    return None


def save_to_cache(cache_filename: str, data: Dict[str, Any]) -> None:
    """Save data to cache."""
    with open(cache_filename, 'w') as f:
        json.dump(data, f, indent=2)


def save_json(filename: str, data: Dict[str, Any]) -> None:
    """Save data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filename: str) -> Dict[str, Any]:
    """Load data from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in a text.
    Uses a simple approximation based on word count.
    """
    # On average, tokens are ~1.3 words in English
    word_count = len(text.split())
    return int(word_count * 1.3)


def estimate_json_token_count(data: Dict[str, Any]) -> int:
    """Estimate the token count for a JSON object."""
    # Serialize to a string and count words
    json_str = json.dumps(data)
    return estimate_token_count(json_str)


def get_timestamp() -> str:
    """Get the current timestamp as ISO format string."""
    return datetime.now().isoformat()


def make_filename_safe(text: str) -> str:
    """Convert text to a safe filename."""
    return text.lower().replace(' ', '_')


def select_items_by_token_budget(items: List[str], scores: List[float],
                                 token_budget: int) -> List[str]:
    """
    Select items based on scores and a token budget.

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