from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

class Document:
    """
    A simple class to mimic document structure with content and metadata.
    """
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}

class RAGTool:
    def __init__(self, name: str, file_paths:list):
        """
        Initialize the RAGTool class with a list of file paths.
        
        :param file_paths: List of paths to the files to be loaded and processed.
        """
        self.name = name
        self.file_paths = file_paths
        self.documents = []
        self.embeddings = HuggingFaceEmbeddings()
        self.db = None
        self.load_files()
    
    def load_files(self):
        """
        Load and process a list of files based on their suffixes.
        
        :param file_paths: List of paths to the files.
        """
        for file_path in self.file_paths:
            if file_path.lower().endswith('.pdf'):
                self.load_pdf(file_path)
            elif file_path.lower().endswith('.txt'):
                self.load_txt(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
        
        # Final reprocessing after all files are loaded
        self._reprocess_documents()
    
    def load_pdf(self, file_path):
        """
        Load and process a single PDF file.
        
        :param file_path: Path to the PDF file.
        """
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        
        print(f"   {self.name} - Splitting into chunks from PDF...")
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(docs)
        print(f"   {self.name} - Number of chunks: {len(texts)}")
        for text in texts:
            print(f"   {self.name} - Content: ", text.page_content.replace('\n', '')[:50])

        print(f"   {self.name} - Adding PDF chunks to the collection...")
        self.documents.extend(texts)
    
    def load_txt(self, file_path):
        """
        Load and process a single text file.
        
        :param file_path: Path to the text file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Wrap text in a Document-like object with metadata
        print(f"   {self.name} - Splitting into chunks from TXT...")
        document = Document(content=text, metadata={'source': file_path})
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents([document])
        print(f"   {self.name} - Number of chunks: {len(texts)}")
        for text in texts:
            print("   {self.name} - Content: ", text.page_content.replace('\n', '')[:50])

        print(f"   {self.name} - Adding TXT chunks to the collection...")
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