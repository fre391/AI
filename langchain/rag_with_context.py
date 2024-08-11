import warnings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

warnings.filterwarnings('ignore')

class Document:
    """
    A simple class to mimic document structure with content and metadata.
    """
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}

class RAG:
    def __init__(self, model: str):
        """
        Initialize the RAG class with a specific model.
        
        :param model: The model name for the language model.
        """
        self.model_name = model
        self.documents = []
        self.chain = None
        self.embeddings = HuggingFaceEmbeddings()
        self.db = None
        self.previous_results = {}  # Store results of previous queries
    
    def load_pdf(self, file_path):
        """
        Load and process a single PDF file.
        
        :param file_path: Path to the PDF file.
        """
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        
        print(f"   Splitting into chunks...")
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(docs)
        print(f"   Number of chunks created : {len(texts)}")
        for text in texts:
            print("     --> ", text.page_content.replace('\n', '')[:50])

        print(f"   Adding to the collection...")
        self.documents.extend(texts)
        
        # Reprocess documents to update the FAISS index
        self._reprocess_documents()
    
    def load_txt(self, file_path):
        """
        Load and process a single text file.
        
        :param file_path: Path to the text file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Wrap text in a Document-like object with metadata
        print(f"   Splitting into chunks...")
        document = Document(content=text, metadata={'source': file_path})
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents([document])
        print(f"   Number of chunks created: {len(texts)}")
        for text in texts:
            print("     --> ", text.page_content.replace('\n', '')[:50])

        print(f"   Adding to the collection...")
        self.documents.extend(texts)
        
        # Reprocess documents to update the FAISS index
        self._reprocess_documents()
    
    def _reprocess_documents(self):
        """
        Recreate the FAISS index and retrieval chain based on the updated documents.
        """
        if not self.documents:
            raise RuntimeError("No documents loaded. Please load at least one document.")
        
        print("   Creating embeddings for document chunks...")
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        
        # Print the number of documents in FAISS index using ntotal
        num_documents = self.db.index.ntotal
        print(f"   Number of chunks in FAISS index: {num_documents}")

        print("   Setting up the retrieval chain...")
        llm = Ollama(model=self.model_name)
        self.chain = RetrievalQA.from_chain_type(
            llm,
            retriever=self.db.as_retriever()
        )

    def query(self, question: str, context: str = None):
        """
        Query the loaded documents with a question, optionally using previous results as context.
        
        :param question: The question to be asked.
        :param context: Optional context to be included in the query.
        :return: The result of the query.
        """
        if not self.chain:
            raise RuntimeError("No documents loaded. Please load documents before querying.")
        
        # Create the final query text
        final_query = f"{context}\n\n{question}" if context else question
        
        print(f"Agent query: {final_query}")
        result = self.chain.invoke({"query": final_query})
        
        # Store the result of the query
        self.previous_results[question] = result['result']
        
        return result['result']

# Example usage:

pdf_paths = [
    # Add more PDF file paths as needed
]

txt_paths = [
    "/Users/markusfreyt/Development/Projects/AI/langchain/docs/TPLINK1.txt",
    "/Users/markusfreyt/Development/Projects/AI/langchain/docs/TPLINK2.txt",
    # Add more text file paths as needed
]

rag = RAG(model="llama3.1")

# Load PDF files
for file_path in pdf_paths:
    print()
    print(f"Loading PDF: {file_path}...")
    rag.load_pdf(file_path)

# Load text files
for file_path in txt_paths:
    print()
    print(f"Loading TXT: {file_path}...")
    rag.load_txt(file_path)

# Query the loaded documents
print()
first_query = "Welcher der beiden TP-Link Router ist preiswerter?"
first_result = rag.query(first_query)
print("--------------------------")
print(first_result)
print("--------------------------")

# Use the result of the first query as context for the second query
print()
second_query = "Welche Infos hat das Gerät aus der vorherigen Antwort?"
context = first_result  # Use the result from the first query as context
second_result = rag.query(second_query, context=context)
print("--------------------------")
print(second_result)
print("--------------------------")
