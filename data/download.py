import requests

def download_fasta(url: str, filename: str):
    """
    Download FASTA data from a URL and save to a file.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'w') as f:
            f.write(response.text)
        print(f"Downloaded and saved: {filename}")
    else:
        raise Exception(f"Failed to download FASTA from {url}")

