# Plant Disease Treatment System

A modular system for plant disease diagnosis, information retrieval, and personalized treatment recommendations.

## Overview

This system helps gardeners identify plant diseases and provides tailored treatment recommendations based on their specific situation. It can:

1. Search for and extract information on plant diseases from trusted sources
2. Process and synthesize disease information into structured data
3. Generate personalized treatment recommendations based on the gardener's specific situation
4. Create comprehensive reports for both disease information and treatment plans

The system can work with an optional local LLM (via LM Studio) to generate highly personalized recommendations, or it can fall back to a rule-based approach if no LLM is available.

## System Architecture

The system is organized into modular components that handle specific tasks in the workflow:

```
├── main.py                   # Main application entry point
├── config.py                 # Configuration settings and constants
├── utils.py                  # Utility functions used across modules
├── web_scraper.py            # Handles searching for and fetching web content
├── content_extractor.py      # Extracts relevant content from HTML
├── disease_parser.py         # Parses content to extract disease information
├── data_processor.py         # Synthesizes results from multiple sources
├── context_analyzer.py       # Analyzes user context for relevance
├── recommendation_generator.py # Generates personalized recommendations
├── report_formatter.py       # Formats information into readable reports
└── user_input.py             # Handles user input collection
```

## Installation

### Requirements

- Python 3.6+
- Required packages (installed automatically):
  - requests
  - beautifulsoup4
  - scikit-learn (optional, for advanced text analysis)
  - googlesearch-python (optional, for web search)

### Optional Components

- [LM Studio](https://lmstudio.ai/) - For generating personalized recommendations using a local LLM

### Installation Steps

1. Clone this repository or download the files

2. Install the required packages:
   ```
   pip install requests beautifulsoup4 scikit-learn
   ```

3. (Optional) Install googlesearch-python for web search functionality:
   ```
   pip install googlesearch-python
   ```

4. (Optional) Set up LM Studio:
   - Download and install [LM Studio](https://lmstudio.ai/)
   - Download a suitable language model (preferably >7B parameters)
   - Start the local API server in LM Studio (on port 1234 by default)

## Usage

### Interactive Mode

The easiest way to use the system is in interactive mode:

```
python main.py --interactive
```

This will guide you through the process of:
1. Specifying the plant and disease (if known)
2. Providing details about your specific situation
3. Generating a personalized recommendation

### Command Line Options

```
python main.py --plant PLANT_NAME [--disease DISEASE_NAME] [--input-file INPUT_FILE] [--output-file OUTPUT_FILE] [--data-dir DATA_DIR] [--lm-studio-url LM_STUDIO_URL]
```

Options:
- `--plant`: Name of the plant (required if not in interactive mode)
- `--disease`: Specific disease name (optional)
- `--input-file`: Path to input file (optional)
- `--output-file`: Output file for the recommendation (optional)
- `--data-dir`: Directory for plant disease data (default: ./plant_diseases)
- `--lm-studio-url`: URL for LM Studio API (default: http://localhost:1234/v1)
- `--interactive`: Run in interactive mode

## Example

```
python main.py --plant tomato --disease "early blight" --output-file tomato_treatment.md
```

This will:
1. Look for existing information about tomato early blight
2. If none exists, search trusted sources on the web
3. Ask for details about your specific situation
4. Generate a personalized treatment recommendation
5. Save the recommendation to tomato_treatment.md

## Output Structure

The system creates a structured directory for each plant:

```
./plant_diseases/
└── tomato/
    ├── cache/                        # Cached web pages
    ├── disease_report.md             # Disease information report
    ├── treatment_recommendation.md   # Personalized recommendation
    └── all_diseases_info.json        # Structured disease data
```

## Extending the System

### Adding Trusted Domains

You can add more trusted domains for information sources in the `config.py` file.

### Improving Text Analysis

The system uses scikit-learn for text relevance analysis when available. Install it for better results:

```
pip install scikit-learn
```

### Using Different LLMs

The system is designed to work with LM Studio by default, but you can modify the `recommendation_generator.py` file to use other LLM APIs.

## License

This project is open source and available under the MIT License.