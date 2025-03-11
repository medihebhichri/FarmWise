import requests
from bs4 import BeautifulSoup
import argparse
import re
import json
import os
from datetime import datetime
from urllib.parse import urlparse


class PlantDiseaseAgent:
    def __init__(self, base_dir="./plant_diseases"):
        """
        Initialize the agent with a base directory for storing results.
        Each plant will have its own subfolder.
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Expanded list of trusted domains for plant disease information
        self.trusted_domains = [
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

    def get_plant_cache_dir(self, plant_name):
        """Get the cache directory for a specific plant."""
        plant_dir = os.path.join(self.base_dir, plant_name.lower().replace(' ', '_'))
        cache_dir = os.path.join(plant_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def search_google(self, query, num_results=10):
        """
        Search for plant disease information.
        Uses the googlesearch-python library if available, otherwise uses mock data.
        """
        try:
            from googlesearch import search
            results = list(search(query, num_results=num_results))
            print(f"Found {len(results)} search results")
            return results
        except ImportError:
            print("Warning: googlesearch-python not installed. Using mock search results.")
            plant_name = query.split()[0].lower()

            # Dynamic mock results based on plant name
            mock_domains = {
                "tomato": [
                    "https://extension.umn.edu/plant-diseases/tomato-diseases",
                    "https://extension.umd.edu/resource/leaf-spots-tomato",
                    "https://ipm.ucanr.edu/agriculture/tomato/",
                    "https://extension.psu.edu/tomato-diseases"
                ],
                "apple": [
                    "https://extension.umn.edu/plant-diseases/apple-scab",
                    "https://extension.psu.edu/apple-diseases",
                    "https://extension.missouri.edu/publications/g6026",
                    "https://extension.unh.edu/blog/2019/05/common-apple-diseases"
                ],
                "rose": [
                    "https://extension.psu.edu/rose-diseases",
                    "https://extension.umn.edu/plant-diseases/rose-diseases",
                    "https://rhs.org.uk/plants/roses/diseases",
                    "https://gardeningknowhow.com/ornamental/flowers/roses/common-rose-diseases"
                ]
            }

            # Default mock results if plant not in predefined list
            default_results = [
                f"https://extension.umn.edu/plant-diseases/{plant_name}-diseases",
                f"https://extension.psu.edu/{plant_name}-diseases",
                f"https://gardeningknowhow.com/plant-problems/{plant_name}-diseases",
                f"https://rhs.org.uk/plants/{plant_name}/diseases"
            ]

            return mock_domains.get(plant_name, default_results)

    def is_trusted_domain(self, url):
        """Check if the URL is from a trusted domain."""
        domain = urlparse(url).netloc
        return any(trusted in domain for trusted in self.trusted_domains)

    def clean_text(self, text):
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

    def extract_content(self, soup, url):
        """Extract relevant content based on the page structure."""
        domain = urlparse(url).netloc
        content = ""

        # Try to find the main content using different selectors
        main_content = None

        # Common content selectors for different website structures
        selectors = [
            'article', 'main', '.content', '#content', '.main-content', '.article',
            '.post-content', '.entry-content', '.page-content', '#main-content',
            '.article-body', '.article-content', '.post', '.entry', '.page',
            'section[role="main"]', 'div[role="main"]', '.container', '.wrapper'
        ]

        # Try each selector
        for selector in selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # Extract text from the main content area if found
        if main_content:
            # First try to get paragraphs
            paragraphs = main_content.find_all('p')
            if paragraphs:
                content = " ".join(p.get_text() for p in paragraphs)
            else:
                # If no paragraphs, get all text
                content = main_content.get_text()
        else:
            # Fallback: get all paragraphs from the page
            paragraphs = soup.find_all('p')
            content = " ".join(p.get_text() for p in paragraphs)

        # Clean the content
        content = self.clean_text(content)

        return content

    def extract_disease_info(self, content, plant_name):
        """
        Extract structured information about plant diseases from content.
        Returns a dictionary with disease information.
        """
        info = {
            "disease_names": [],
            "symptoms": [],
            "causes": [],
            "treatments": [],
            "prevention": []
        }

        # Extract disease names (looking for disease mentions)
        # Improved pattern to catch more disease names
        disease_patterns = [
            r'([A-Z][a-z]+(?: [a-z]+)?(?: [Dd]isease| [Bb]light| [Rr]ot| [Mm]ildew| [Ss]pot| [Ww]ilt))',
            r'(' + re.escape(plant_name.capitalize()) + r' (leaf spot|scab|rust|canker|mosaic|blight|rot|mildew|wilt))',
            r'([A-Z][a-z]+ (leaf spot|scab|rust|canker|mosaic virus|blight|rot|mildew|wilt))'
        ]

        all_diseases = []
        for pattern in disease_patterns:
            matches = re.findall(pattern, content)
            if isinstance(matches[0], tuple) if matches else False:
                # Extract the full match if the result is a tuple of groups
                matches = [match[0] for match in matches]
            all_diseases.extend(matches)

        if all_diseases:
            info["disease_names"] = list(set(all_diseases))

        # Enhanced symptom extraction
        symptom_patterns = [
            r'(?:Symptoms|Signs)[:\s]+(.*?)(?=\b(?:Causes|Treatment|Prevention|Control|Management)\b|$)',
            r'(?:Common symptoms|Characteristic symptoms)[:\s]+(.*?)(?=\b(?:Causes|Treatment|Prevention|Control|Management)\b|$)',
            r'(?:Symptoms include|Signs include)[:\s]+(.*?)(?=\b(?:Causes|Treatment|Prevention|Control|Management)\b|$)'
        ]

        for pattern in symptom_patterns:
            sections = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if sections:
                # Split into sentences and add to symptoms
                symptoms = re.split(r'(?<=[.!?])\s+(?=[A-Z])', sections[0])
                info["symptoms"].extend([s.strip() for s in symptoms if s.strip()])

        # Enhanced cause extraction
        cause_patterns = [
            r'(?:Causes|Pathogen|Causal agent|Disease agent)[:\s]+(.*?)(?=\b(?:Symptoms|Treatment|Prevention|Control|Management)\b|$)',
            r'(?:The disease is caused by|This disease is caused by)[:\s]+(.*?)(?=\b(?:Symptoms|Treatment|Prevention|Control|Management)\b|$)',
            r'(?:Caused by)[:\s]+(.*?)(?=\b(?:Symptoms|Treatment|Prevention|Control|Management)\b|$)'
        ]

        for pattern in cause_patterns:
            sections = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if sections:
                causes = re.split(r'(?<=[.!?])\s+(?=[A-Z])', sections[0])
                info["causes"].extend([c.strip() for c in causes if c.strip()])

        # Enhanced treatment extraction
        treatment_patterns = [
            r'(?:Treatment|Control|Management|Control measures)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Prevention)\b|$)',
            r'(?:How to treat|How to control|How to manage)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Prevention)\b|$)',
            r'(?:Treatment options|Control options|Management strategies)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Prevention)\b|$)'
        ]

        for pattern in treatment_patterns:
            sections = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if sections:
                treatments = re.split(r'(?<=[.!?])\s+(?=[A-Z])', sections[0])
                info["treatments"].extend([t.strip() for t in treatments if t.strip()])

        # Enhanced prevention extraction
        prevention_patterns = [
            r'(?:Prevention|Avoid|Preventing|Prevention measures)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Treatment|Control|Management)\b|$)',
            r'(?:How to prevent|Ways to prevent|Prevention strategies)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Treatment|Control|Management)\b|$)',
            r'(?:To prevent|For prevention)[:\s]+(.*?)(?=\b(?:Symptoms|Causes|Treatment|Control|Management)\b|$)'
        ]

        for pattern in prevention_patterns:
            sections = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if sections:
                preventions = re.split(r'(?<=[.!?])\s+(?=[A-Z])', sections[0])
                info["prevention"].extend([p.strip() for p in preventions if p.strip()])

        return info

    def fetch_and_parse_url(self, url, plant_name):
        """Fetch and parse content from a URL."""
        try:
            # Get the cache directory for this plant
            cache_dir = self.get_plant_cache_dir(plant_name)

            # Check cache first
            cache_filename = os.path.join(cache_dir, f"{urlparse(url).netloc}_{hash(url)}.json")
            if os.path.exists(cache_filename):
                with open(cache_filename, 'r') as f:
                    cached_data = json.load(f)
                print(f"Retrieved from cache: {url}")
                return cached_data

            # Send request with headers
            print(f"Fetching: {url}")
            response = requests.get(url, headers=self.headers, timeout=15)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract the page title
                title = soup.title.string if soup.title else "No Title"

                # Extract content
                content = self.extract_content(soup, url)

                # Extract structured disease information
                disease_info = self.extract_disease_info(content, plant_name)

                # Prepare result
                result = {
                    "url": url,
                    "title": title,
                    "content_snippet": content[:300] + "..." if len(content) > 300 else content,
                    "disease_info": disease_info,
                    "timestamp": datetime.now().isoformat()
                }

                # Cache the result
                with open(cache_filename, 'w') as f:
                    json.dump(result, f, indent=2)

                return result
            else:
                print(f"Failed to retrieve content from {url}, status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return None

    def search_plant_disease(self, plant_name, disease_name=None):
        plant_dir = os.path.join(self.base_dir, plant_name.lower().replace(' ', '_'))
        os.makedirs(plant_dir, exist_ok=True)

        if disease_name:
            query = f"{plant_name} {disease_name} disease symptoms treatment"
        else:
            query = f"{plant_name} common diseases symptoms treatment prevention"

        print(f"Searching for: {query}")

        search_results = self.search_google(query, num_results=15)

        trusted_results = [url for url in search_results if self.is_trusted_domain(url)]
        if not trusted_results:
            trusted_results = search_results[:5]  # Take first 5 if no trusted domains
            print("Warning: No trusted domains found in search results. Using top 5 results.")
        else:
            print(f"Found {len(trusted_results)} results from trusted domains.")

        all_results = []
        for url in trusted_results[:8]:  # Limit to first 8 trusted results
            result = self.fetch_and_parse_url(url, plant_name)
            if result:
                all_results.append(result)

        return self.synthesize_results(all_results, plant_name, disease_name)

    def synthesize_results(self, results, plant_name, disease_name=None):
        """Synthesize results from multiple sources into a cohesive summary."""
        if not results:
            return {
                "status": "error",
                "message": f"No information found for {plant_name} diseases."
            }

        combined_info = {
            "plant": plant_name,
            "diseases": [],
            "sources": [{"url": r["url"], "title": r["title"]} for r in results]
        }

        if disease_name:
            disease_data = {
                "name": disease_name,
                "symptoms": [],
                "causes": [],
                "treatments": [],
                "prevention": []
            }

            for result in results:
                info = result["disease_info"]
                disease_data["symptoms"].extend(info["symptoms"])
                disease_data["causes"].extend(info["causes"])
                disease_data["treatments"].extend(info["treatments"])
                disease_data["prevention"].extend(info["prevention"])

            for key in ["symptoms", "causes", "treatments", "prevention"]:
                disease_data[key] = list(set(disease_data[key]))

            combined_info["diseases"].append(disease_data)
        else:
            all_diseases = {}

            for result in results:
                disease_names = result["disease_info"]["disease_names"]

                if not disease_names:
                    disease_names = ["General Disease Information"]

                for name in disease_names:
                    if name not in all_diseases:
                        all_diseases[name] = {
                            "name": name,
                            "symptoms": [],
                            "causes": [],
                            "treatments": [],
                            "prevention": []
                        }
                    info = result["disease_info"]
                    all_diseases[name]["symptoms"].extend(info["symptoms"])
                    all_diseases[name]["causes"].extend(info["causes"])
                    all_diseases[name]["treatments"].extend(info["treatments"])
                    all_diseases[name]["prevention"].extend(info["prevention"])

            for name, data in all_diseases.items():
                for key in ["symptoms", "causes", "treatments", "prevention"]:
                    data[key] = list(set(data[key]))
                combined_info["diseases"].append(data)

        plant_dir = os.path.join(self.base_dir, plant_name.lower().replace(' ', '_'))
        if disease_name:
            json_filename = os.path.join(plant_dir, f"{disease_name.lower().replace(' ', '_')}_info.json")
        else:
            json_filename = os.path.join(plant_dir, "all_diseases_info.json")

        with open(json_filename, 'w') as f:
            json.dump(combined_info, f, indent=2)

        return combined_info

    def generate_report(self, combined_info):
        plant_name = combined_info["plant"]
        diseases = combined_info["diseases"]
        sources = combined_info["sources"]

        plant_dir = os.path.join(self.base_dir, plant_name.lower().replace(' ', '_'))
        if len(diseases) == 1 and diseases[0]["name"] != "General Disease Information":
            report_filename = os.path.join(plant_dir, f"{diseases[0]['name'].lower().replace(' ', '_')}_report.md")
        else:
            report_filename = os.path.join(plant_dir, "disease_report.md")

        report = f"# Plant Disease Report: {plant_name.capitalize()}\n\n"
        report += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

        if not diseases:
            report += "No specific disease information found.\n"
        else:

            if len(diseases) > 1:
                report += "## Table of Contents\n\n"
                for i, disease in enumerate(diseases, 1):
                    report += f"{i}. [{disease['name']}](#{disease['name'].lower().replace(' ', '-')})\n"
                report += "\n"

            for disease in diseases:
                disease_anchor = disease['name'].lower().replace(' ', '-')
                report += f"## {disease['name']}\n\n"

                if disease["symptoms"]:
                    report += "### Symptoms\n\n"
                    for symptom in disease["symptoms"]:
                        report += f"- {symptom}\n"
                    report += "\n"

                if disease["causes"]:
                    report += "### Causes\n\n"
                    for cause in disease["causes"]:
                        report += f"- {cause}\n"
                    report += "\n"

                if disease["treatments"]:
                    report += "### Treatment\n\n"
                    for treatment in disease["treatments"]:
                        report += f"- {treatment}\n"
                    report += "\n"

                if disease["prevention"]:
                    report += "### Prevention\n\n"
                    for prevention in disease["prevention"]:
                        report += f"- {prevention}\n"
                    report += "\n"

        report += "## Sources\n\n"
        for source in sources:
            report += f"- [{source['title']}]({source['url']})\n"

        with open(report_filename, 'w') as f:
            f.write(report)

        print(f"Report written to {report_filename}")
        return report_filename


def main():

    parser = argparse.ArgumentParser(description='Search for plant disease information')
    parser.add_argument('plant', help='Name of the plant')
    parser.add_argument('--disease', help='Specific disease name (optional)')
    parser.add_argument('--output-dir', help='Base output directory (optional)', default='./plant_diseases')

    args = parser.parse_args()

    # Create agent
    agent = PlantDiseaseAgent(base_dir=args.output_dir)

    # Search for information
    results = agent.search_plant_disease(args.plant, args.disease)

    # Generate report
    report_file = agent.generate_report(results)

    print(f"\nSearch complete! Results are organized in the following directory structure:")
    print(f"{args.output_dir}/")
    print(f"└── {args.plant.lower().replace(' ', '_')}/")
    print(f"    ├── cache/                  # Cached web pages")
    print(f"    ├── disease_report.md       # Markdown report")
    if args.disease:
        print(f"    └── {args.disease.lower().replace(' ', '_')}_info.json  # Structured data")
    else:
        print(f"    └── all_diseases_info.json  # Structured data")


if __name__ == "__main__":
    main()