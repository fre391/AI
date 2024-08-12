import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

class Document:
    """
    A simple class to mimic document structure with content and metadata.
    """
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}

class WebTool:
    def __init__(self, name: str, urls: list):
        """
        Initialize the WebTool class with a list of URLs.
        
        :param name: Name of the WebTool instance.
        :param urls: List of URLs to be loaded and processed.
        """
        self.name = name
        self.urls = urls
        self.documents = []
        self.embeddings = HuggingFaceEmbeddings()
        self.db = None
        
        self.load_webpages()
    
    def load_webpages(self):
        """
        Load and process a list of webpages based on the provided URLs.
        """
        for url in self.urls:
            self.load_webpage(url)
        
        # Final reprocessing after all webpages are loaded
        self._reprocess_documents()
    
    def load_webpage(self, url):
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
        
        # Wrap text in a Document-like object with metadata
        print(f"   {self.name} - Splitting into chunks from Webpage: {url}...")
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