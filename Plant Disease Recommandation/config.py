import os

# Base directory for storing results
DEFAULT_BASE_DIR = "./plant_diseases"

# Default LM Studio URL
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"

# Default token limits
DEFAULT_TOKEN_LIMIT = 2048
MODEL_TOKEN_LIMITS = {
    "70b": 8192,
    "13b": 4096,
    "15b": 4096,
    "7b": 4096,
    "8b": 4096,
    "3.5-sonnet": 8192,
    "3.5-haiku": 4096,
    "3-opus": 16384
}

# Headers for web requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# List of trusted domains for plant disease information
TRUSTED_DOMAINS = [
    # University Extensions
    "extension.umn.edu", "extension.umd.edu", "extension.psu.edu",
    "extension.cornell.edu", "ipm.ucanr.edu", "aces.illinois.edu",
    "hgic.clemson.edu", "ag.umass.edu", "hort.extension.wisc.edu",
    "extension.missouri.edu", "extension.purdue.edu", "extension.colostate.edu",
    "extension.oregonstate.edu", "extension.wsu.edu", "extension.uga.edu",
    "extension.ufl.edu", "extension.tennessee.edu", "extension.msstate.edu",
    "extension.unh.edu", "extension.okstate.edu", "extension.usu.edu",
    "extension.arizona.edu", "extension.unr.edu", "extension.unl.edu",
    "extension.iastate.edu", "extension.uark.edu", "extension.ucdavis.edu",

    # Government Resources
    "usda.gov", "ars.usda.gov", "nifa.usda.gov", "aphis.usda.gov",

    # Horticultural Organizations
    "rhs.org.uk", "missouribotanicalgarden.org", "chicagobotanic.org",
    "nybg.org", "arborday.org", "bbg.org", "plantclinic.cornell.edu",

    # Gardening Websites
    "gardeningknowhow.com", "almanac.com", "gardeners.com",
    "garden.org", "growveg.com", "finegardening.com", "bhg.com",
    "groworganic.com", "davesgarden.com", "balconygardenweb.com",

    # University Plant Pathology Departments
    "plantpath.cornell.edu", "plantpath.wisc.edu", "plantpath.ifas.ufl.edu",
    "plantpathology.ces.ncsu.edu", "plant-pest-advisory.rutgers.edu",

    # Agricultural Research Centers
    "oardc.osu.edu", "nysipm.cornell.edu", "ipm.ucanr.edu",
    "agsci.psu.edu", "agrilife.org"
]

# Text patterns for extracting information
DISEASE_PATTERNS = [
    r'([A-Z][a-z]+(?: [a-z]+)?(?: [Dd]isease| [Bb]light| [Rr]ot| [Mm]ildew| [Ss]pot| [Ww]ilt))',
    r'({plant_name} (leaf spot|scab|rust|canker|mosaic|blight|rot|mildew|wilt))',
    r'([A-Z][a-z]+ (leaf spot|scab|rust|canker|mosaic virus|blight|rot|mildew|wilt))'
]

SYMPTOM_PATTERNS = [
    r'(?:Symptoms|Signs)[:\s]+(.*?)(?=\b(?:Causes|Treatment|Prevention|Control|Management)\b|$)',
    r'(?:Common symptoms|Characteristic symptoms)[:\s]+(.*?)(?=\b(?:Causes|Treatment|Prevention|Control|Management)\b|$)',
    r'(?:Symptoms include|Signs include)[:\s]+(.*?)(?=\b(?:Causes|Treatment|Prevention|Control|Management)\b|$)'
]

CAUSE_PATTERNS = [
    r'(?:Causes|Pathogen|Causal agent|Disease agent)[:\s]+(.*?)(?=\b(?:Symptoms|Treatment|Prevention|Control|Management)\b|$)',
    r'(?:The disease is caused by|This disease is caused by)[:\s]+(.*?)(?=\b(?:Symptoms|Treatment|Prevention|Control|Management)\b|$)',
    r'(?:Caused by)[:\s]+(.*?)(?=\b(?:Symptoms|Treatment|Prevention|Control|Management)\b|$)'
]

TREATMENT_PATTERNS = [
    r'(?:Treatment|Control|Management|Control measures)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Prevention)\b|$)',
    r'(?:How to treat|How to control|How to manage)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Prevention)\b|$)',
    r'(?:Treatment options|Control options|Management strategies)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Prevention)\b|$)'
]

PREVENTION_PATTERNS = [
    r'(?:Prevention|Avoid|Preventing|Prevention measures)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Treatment|Control|Management)\b|$)',
    r'(?:How to prevent|Ways to prevent|Prevention strategies)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Treatment|Control|Management)\b|$)',
    r'(?:To prevent|For prevention)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Treatment|Control|Management)\b|$)'
]

# Fallback organic and conventional treatments
ORGANIC_TREATMENTS = [
    "Apply neem oil (2 tbsp per gallon of water) to all affected parts weekly for 3 weeks.",
    "Use a copper-based organic fungicide labeled for fruit trees.",
    "Apply compost tea as a foliar spray to boost plant immunity.",
    "Spray with a solution of 1 tablespoon baking soda, 1 teaspoon mild soap, and 1 gallon of water."
]

CONVENTIONAL_TREATMENTS = [
    "Apply a copper-based fungicide according to package directions.",
    "Use a systemic fungicide containing chlorothalonil for more severe cases.",
    "Apply a broad-spectrum fungicide that targets this specific pathogen.",
    "Treat with appropriate bactericide if bacterial infection is confirmed."
]

# Category weights for token budgeting
CATEGORY_WEIGHTS = {
    "symptoms": 0.2,      # 20% for symptoms
    "causes": 0.15,       # 15% for causes
    "treatments": 0.4,    # 40% for treatments (most important)
    "prevention": 0.25    # 25% for prevention
}

# Common HTML content selectors for web scraping
HTML_CONTENT_SELECTORS = [
    'article', 'main', '.content', '#content', '.main-content', '.article',
    '.post-content', '.entry-content', '.page-content', '#main-content',
    '.article-body', '.article-content', '.post', '.entry', '.page',
    'section[role="main"]', 'div[role="main"]', '.container', '.wrapper'
]

# System prompt for LM Studio
LM_SYSTEM_PROMPT = "You are a plant disease expert providing practical treatment recommendations."