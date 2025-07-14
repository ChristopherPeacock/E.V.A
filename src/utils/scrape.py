import requests
from bs4 import BeautifulSoup

def scrape_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = soup.get_text()
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join([l for l in lines if l])
