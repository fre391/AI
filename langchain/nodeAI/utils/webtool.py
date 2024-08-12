import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import CharacterTextSplitter
from urllib.parse import urljoin, urlparse
from collections import deque
import time

from .tool import Tool, Document

class WebTool(Tool):
    def __init__(self, name: str, urls: list, follow_redirects: bool = False, max_depth: int = 0):
        """
        Initialize the WebTool class with a list of URLs and options for subpages and depth limit.
        
        :param name: Name of the WebTool instance.
        :param urls: List of URLs to be loaded and processed.
        :param subpages: Boolean to specify whether to include subpages.
        :param max_depth: Maximum depth to crawl the website.
        """
        super().__init__(name, urls) 
        self.follow_redirects = follow_redirects
        self.max_depth = max_depth
        self.base_domain = self._get_base_domain(urls[0])
        self.load()
    
    def load(self):
        """
        Collect all pages including subpages from the starting URLs up to a specified depth.
        """
        #self.sources = set()
        queue = deque([(url, 0) for url in self.sources])

        self.sources = []
        
        while queue:
            url, depth = queue.popleft()
            if url in self.sources or depth > self.max_depth:
                continue

            self.add_source(url)
            #self.sources.add(url)
            self.load_webpage(url, depth)
            
            if self.follow_redirects and depth < self.max_depth:
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                except (requests.RequestException, requests.HTTPError) as e:
                    print(f"Failed to retrieve the webpage {url}: {e}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                for a_tag in soup.find_all('a', href=True):
                    link = a_tag['href']
                    full_url = urljoin(url, link)
                    parsed_url = urlparse(full_url)
                    
                    # Normalize and check if the URL is within the same base domain
                    if parsed_url.netloc.lower() == self.base_domain and full_url not in self.sources:
                        # Check if the URL points to an HTML page or directory
                        if self._is_html_or_directory(full_url):
                            queue.append((full_url, depth + 1))
                
                time.sleep(1)
        super().load()


    def load_webpages(self, urls):
        """
        Load and process a list of webpages based on the provided URLs.
        
        :param urls: List of URLs to be fetched and processed.
        """
        for url in urls:
            self.load_webpage(url, depth=0)
            
        self._reprocess_documents()
    
    def load_webpage(self, url, depth):
        """
        Load and process a webpage.
        
        :param url: URL of the webpage to be fetched.
        """
        print(f"   {self.name} - Loading {url} ({depth} / {self.max_depth})")

        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve the webpage {url}: {response.status_code}")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        if (self.debug_output): 
            print(f"   {self.name} - Splitting into chunks...")
        document = Document(content=text, metadata={'source': url})
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents([document])
        if (self.debug_output): 
            print(f"   {self.name} - Number of chunks: {len(texts)}")
            for text in texts:
                print(f"   {self.name} - Content: ", text.page_content.replace('\n', '')[:50])

            print(f"   {self.name} - Adding Webpage chunks to the collection...")

        self.documents.extend(texts)

    def _get_base_domain(self, url):
        """
        Extract and normalize the base domain from a URL.
        
        :param url: URL to extract the base domain from.
        :return: Normalized base domain.
        """
        parsed_url = urlparse(url)
        return parsed_url.netloc.lower()
    
    def _is_html_or_directory(self, url):
        """
        Check if the URL points to an HTML page or directory.
        
        :param url: URL to check.
        :return: Boolean indicating if the URL points to an HTML page or directory.
        """
        parsed_url = urlparse(url)
        path = parsed_url.path
        if path.endswith('/') or path.endswith('.html') or path.endswith('.htm'):
            return True
        return False