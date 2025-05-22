
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "plant_data")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
for dir_path in [DATA_DIR, VECTOR_DB_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# API Keys and service endpoints
YOUTUBE_API_KEY = 'AIzaSyA6IIgizjfr7O66rI-KeDOdjlm_BkdvFV4'
LLM_SERVER_URL = "http://localhost:1234"
COMPLETIONS_ENDPOINT = f"{LLM_SERVER_URL}/v1/completions"
DEFAULT_MODEL = "your-model-name"  # Update with your model name

# Embedding model settings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

# Language and dialect support
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "dialects": ["US", "UK", "AU", "CA"],
        "active": True
    },
    "ar": {
        "name": "Arabic",
        "dialects": ["EG", "SA", "MA", "TN", "DZ", "LB", "JO", "IQ", "SY"],
        # EG: Egyptian, SA: Saudi, MA: Moroccan, TN: Tunisian, DZ: Algerian,
        # LB: Lebanese, JO: Jordanian, IQ: Iraqi, SY: Syrian
        "active": True
    },
    "fr": {
        "name": "French",
        "dialects": ["FR", "CA", "BE", "CH", "MA", "TN"],
        # FR: France, CA: Canada, BE: Belgium, CH: Switzerland,
        # MA: Morocco, TN: Tunisia (French dialects spoken in North Africa)
        "active": True
    },
    "es": {
        "name": "Spanish",
        "dialects": ["ES", "MX", "AR", "CO", "CL", "PE"],
        "active": True
    },
    "de": {
        "name": "German",
        "dialects": ["DE", "AT", "CH"],
        "active": True
    },
    "it": {
        "name": "Italian",
        "dialects": ["IT", "CH"],
        "active": True
    },
    "pt": {
        "name": "Portuguese",
        "dialects": ["PT", "BR"],
        "active": True
    },
    "ru": {
        "name": "Russian",
        "dialects": ["RU", "BY", "KZ"],
        "active": True
    },
    "zh": {
        "name": "Chinese",
        "dialects": ["CN", "TW", "HK", "SG"],
        "active": True
    },
    "ja": {
        "name": "Japanese",
        "dialects": ["JP"],
        "active": True
    },
    "hi": {
        "name": "Hindi",
        "dialects": ["IN"],
        "active": True
    },
    "tr": {
        "name": "Turkish",
        "dialects": ["TR"],
        "active": True
    }
}

# Language code mappings for services
YOUTUBE_LANGUAGE_MAPPING = {
    "zh": "zh-CN",  # Map Chinese to Simplified Chinese for YouTube
    "ar": "ar"      # Keep Arabic as is for YouTube
}

# Dialect-specific language adjustments (for specialized querying and processing)
DIALECT_ADJUSTMENTS = {
    "ar_TN": {
        "query_terms": ["نباتات", "زراعة", "بستنة", "فلاحة تونسية", "زراعة منزلية", "عناية بالنباتات"],
        "common_phrases": ["كيفاش نزرع", "كيفاش نعتني ب", "طريقة زراعة", "الفلاحة التونسية"]
    },
    "ar_EG": {
        "query_terms": ["نباتات", "زراعة", "بستنة", "فلاحة مصرية", "زراعة منزلية"],
        "common_phrases": ["ازاي ازرع", "طريقة زراعة", "العناية بالنباتات"]
    },
    "fr_TN": {
        "query_terms": ["plantes", "jardinage", "culture", "agriculture tunisienne"],
        "common_phrases": ["comment cultiver", "jardinage en Tunisie", "plantes d'intérieur"]
    }
}

# Define standard growth factors (consistent across all languages)
GROWTH_FACTOR_KEYS = [
    "temperature",
    "humidity",
    "soil_type",
    "planting_method",
    "care_routine",
    "light_exposure",
    "water_requirements",
    "fertilizer",
    "pH_level",
    "plant_spacing"
]

# Default growth factors (used when extraction fails)
DEFAULT_GROWTH_FACTORS = {key: "Not specified" for key in GROWTH_FACTOR_KEYS}

# Environmental factors that users can input
USER_ENV_FACTORS = [
    "temperature",
    "humidity",
    "soil_humidity",
    "wind_exposure",
    "rainfall",
    "sunlight_hours",
    "location_type",  # indoor or outdoor
    "season"
]

# Trusted gardening websites by language
TRUSTED_WEBSITES = {
    "en": [
        "www.gardeningknowhow.com",
        "www.thespruce.com",
        "www.almanac.com",
        "www.rhs.org.uk",
        "www.gardeners.com"
    ],
    "ar": [
        "www.agri2day.com",
        "www.nabataty.com",
        "www.zira3a.net",
        "www.ehabweb.net",
        "www.agriculture.com.tn"  # Tunisian agricultural website
    ],
    "fr": [
        "www.jardipartage.fr",
        "www.jardiner-malin.fr",
        "www.gerbeaud.com",
        "www.rustica.fr",
        "www.aujardin.info",
        "www.jardinage.ooreka.fr"
    ],
    "es": [
        "www.jardineriaon.com",
        "www.planetahuerto.es",
        "www.infojardin.com",
        "www.jardineriayjardines.com"
    ],
    # Default if language not supported
    "default": [
        "www.gardeningknowhow.com",
        "www.thespruce.com"
    ]
}

# Translation service settings (if using external API)
TRANSLATION_API_KEY = "your-translation-api-key"  # Replace with actual key if using
USE_TRANSLATION_SERVICE = False  # Set to True if using external translation API