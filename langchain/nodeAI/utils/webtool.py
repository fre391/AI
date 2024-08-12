import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from urllib.parse import urljoin, urlparse
from collections import deque
import time

class Document:
    """
    A simple class to mimic document structure with content and metadata.
    """
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}

class WebTool:
    def __init__(self, name: str, urls: list, subpages: bool = False, max_depth: int = 3):
        """
        Initialize the WebTool class with a list of URLs and options for subpages and depth limit.
        
        :param name: Name of the WebTool instance.
        :param urls: List of URLs to be loaded and processed.
        :param subpages: Boolean to specify whether to include subpages.
        :param max_depth: Maximum depth to crawl the website.
        """
        self.name = name
        self.urls = urls
        self.subpages = subpages
        self.max_depth = max_depth
        self.documents = []
        self.embeddings = HuggingFaceEmbeddings()
        self.db = None
        
        # Normalize and initialize the base domain
        self.base_domain = self._get_base_domain(urls[0])
        
        if subpages:
            self._collect_all_pages()
        else:
            self.load_webpages(self.urls)
    
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
    
    def _collect_all_pages(self):
        """
        Collect all pages including subpages from the starting URLs up to a specified depth.
        """
        visited = set()
        queue = deque([(url, 0) for url in self.urls])
        
        while queue:
            url, depth = queue.popleft()
            if url in visited or depth > self.max_depth:
                continue
      
            visited.add(url)
            self.load_webpage(url, depth)
            
            if self.subpages and depth < self.max_depth:
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
                    if parsed_url.netloc.lower() == self.base_domain and full_url not in visited:
                        # Check if the URL points to an HTML page or directory
                        if self._is_html_or_directory(full_url):
                            queue.append((full_url, depth + 1))
                
                time.sleep(1)
        
        self._reprocess_documents()
    
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
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve the webpage {url}: {response.status_code}")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        print()
        print(f"   {self.name} - Fetching Webpage: {url} ({depth} / {self.max_depth})")
        print(f"   {self.name} - Splitting into chunks...")
        document = Document(content=text, metadata={'source': url})
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents([document])
        print(f"   {self.name} - Number of chunks: {len(texts)}")
        for text in texts:
            print(f"   {self.name} - Content: ", text.page_content.replace('\n', '')[:50])

        print(f"   {self.name} - Adding Webpage chunks to the collection...")
        self.documents.extend(texts)
    
    def _reprocess_documents(self):
        """
        Recreate the FAISS index based on the updated documents.
        """
        if not self.documents:
            raise RuntimeError("No documents loaded. Please load at least one document.")
        
        print(f"   {self.name} - Creating embeddings for document chunks...")
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        
        # Print the number of documents in FAISS index using ntotal
        num_documents = self.db.index.ntotal
        print(f"   {self.name} - Number of chunks in FAISS index: {num_documents}")

    def get_document_retriever(self):
        """
        Get the document retriever from the FAISS index.
        
        :return: FAISS retriever
        """
        if not self.db:
            raise RuntimeError("No documents have been processed. Please load documents before getting retriever.")
        return self.db.as_retriever()
