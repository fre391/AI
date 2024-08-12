from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import CharacterTextSplitter

from .tool import Tool, Document

class RAGTool(Tool):
    def __init__(self, name: str, file_paths:list):
        """
        Initialize the RAGTool class with a list of file paths.
        
        :param file_paths: List of paths to the files to be loaded and processed.
        """
        super().__init__(name, file_paths) 
        self.load()
    
    def load(self):
        """
        Load and process a list of files based on their suffixes.
        
        :param file_paths: List of paths to the files.
        """
        for source in self.sources:
            print(f"Loading {source}...")
            if source.lower().endswith('.pdf'):
                self.load_pdf(source)
            elif source.lower().endswith('.txt'):
                self.load_txt(source)
            else:
                print(f"Unsupported file type: {source}")
            print()

        super().load()
    
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
            print(f"   {self.name} - Content: ", text.page_content.replace('\n', '')[:50])

        print(f"   {self.name} - Adding TXT chunks to the collection...")
        self.documents.extend(texts)
    