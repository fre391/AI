from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class Document:
    """
    A simple class to mimic document structure with content and metadata.
    """
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}

class Tool:
    def __init__(self, name:str, sources:list):
        self.debug_output = False
        self.name = name
        self.sources = sources
        self.embeddings = HuggingFaceEmbeddings()
        self.db = None
        self.documents = []

    def get_sources(self):
        return self.sources

    def add_source(self, source):
        if source not in self.sources:
            self.sources.append(source)

    def load(self):
        self._reprocess_documents()

    def _reprocess_documents(self):
        """
        Recreate the FAISS index based on the updated documents.
        """
        if not self.documents:
            raise RuntimeError("No documents loaded. Please load at least one document.")
        
        if (self.debug_output): print(f"   {self.name} - Creating embeddings for document chunks...")
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        
        # Print the number of documents in FAISS index using ntotal
        num_documents = self.db.index.ntotal
        if (self.debug_output): print(f"   {self.name} - Number of chunks in FAISS index: {num_documents}")

    def get_document_retriever(self):
        """
        Get the document retriever from the FAISS index.
        
        :return: FAISS retriever
        """
        if not self.db:
            raise RuntimeError("No documents have been processed. Please load documents before getting retriever.")
        return self.db.as_retriever()