import os
import requests
from dotenv import load_dotenv
load_dotenv()
google_seach_api_key = os.environ["GOOGLE_SEARCH_API_KEY"]


def perform_search(query, api_key):
    search_url = f"https://customsearch.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': api_key,
    }
    
    response = requests.get(search_url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve search results: {response.status_code}")
        return None

# Example usage
if __name__ == "__main__":
    query = 'latest news on AI'

    results = perform_search(query, google_seach_api_key)
    
    if results:
        items = results.get('items', [])
        for item in items:
            print(f"Title: {item['title']}")
            print(f"Link: {item['link']}")
            print(f"Snippet: {item['snippet']}")
            print("-----")