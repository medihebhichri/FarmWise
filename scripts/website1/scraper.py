import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# URL of the citation page
CITATION_URL = "https://orgprints.org/id/eprint/50811/"

# General directory to save PDFs (relative path)
SAVE_DIR = "downloads"
os.makedirs(SAVE_DIR, exist_ok=True)

# Create a session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# Function to extract PDF link
def extract_pdf_link(url):
    print("[*] Accessing citation page...")

    # Wait before request (helps with slow connections)
    time.sleep(5)

    response = session.get(url, timeout=10)
    if response.status_code == 200:
        print("[✓] Page loaded successfully.")
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_link = soup.find('a', class_='ep_document_link')
        if pdf_link:
            return urljoin(url, pdf_link['href'])
    
    print("[✗] No PDF link found.")
    return None

# Function to download the PDF
def download_pdf(pdf_url):
    file_name = os.path.join(SAVE_DIR, pdf_url.split("/")[-1])

    print(f"[*] Downloading PDF to {file_name} ...")

    response = session.get(pdf_url, timeout=20, stream=True)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"[✓] Downloaded: {file_name}")
    else:
        print(f"[✗] Failed to download from {pdf_url}")

# Main function
def main():
    pdf_url = extract_pdf_link(CITATION_URL)
    if pdf_url:
        download_pdf(pdf_url)

if __name__ == "__main__":
    main()

