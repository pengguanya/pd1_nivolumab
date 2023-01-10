import requests

def download_fasta(url: str, filename: str):
    """
    Download FASTA data from a URL and save to a file.
    Added a timeout and custom headers to help mimic a browser.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises HTTPError if the HTTP request returned an unsuccessful status code
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download FASTA from {url}. Error: {e}")
    
    with open(filename, 'w') as f:
        f.write(response.text)
    print(f"Downloaded and saved: {filename}")
