import requests
from bs4 import BeautifulSoup
import json

# Read URLs from file
with open('urls.txt', 'r') as f:
    urls = [line.strip() for line in f if line.strip()]

all_content = []

for url in urls:
    entry = {"url": url}
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract headers and their content
        content_dict = {}
        headers_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if headers_tags:
            for i, header in enumerate(headers_tags):
                header_text = header.get_text(strip=True)
                # Get all text until the next header
                next_header = headers_tags[i+1] if i+1 < len(headers_tags) else None
                content = []
                for sibling in header.next_siblings:
                    if sibling == next_header:
                        break
                    if hasattr(sibling, 'get_text'):
                        text = sibling.get_text(strip=True)
                        if text:
                            content.append(text)
                content_dict[header_text] = '\n'.join(content)
            entry["content"] = content_dict
        else:
            # If no headers, put all text under 'body'
            text = soup.get_text(separator='\n', strip=True)
            entry["content"] = {"body": text}
    except Exception as e:
        entry["error"] = str(e)
    all_content.append(entry)

# Write all content to a single file in JSON format
with open('output.txt', 'w', encoding='utf-8') as f:
    json.dump(all_content, f, ensure_ascii=False, indent=2)

print("Scraping complete. Check output.txt for results.")