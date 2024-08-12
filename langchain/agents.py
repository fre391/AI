import os
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
            print("   {self.name} - Content: ", text.page_content.replace('\n', '')[:50])

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

class Agent:
    def __init__(self, name: str, model: str, rag_tool: RAGTool = None):
        """
        Initialize the Agent class with a specific model and optionally with RAGTool for document processing.
        
        :param model: The model name for the language model.
        :param rag_tool: An optional RAGTool instance for document processing.
        """
        self.agent_name = name
        self.model_name = model
        self.rag_tool = rag_tool
        self.chain = None
        self.previous_results = {}  # Store results of previous queries
    
    def setup_chain(self):
        """
        Set up the retrieval chain using the document retriever from RAGTool if provided.
        """
        llm = Ollama(model=self.model_name)
        if self.rag_tool:
            retriever = self.rag_tool.get_document_retriever()
            self.chain = RetrievalQA.from_chain_type(llm,retriever=retriever)
        else:
            retriever = None
            self.chain = llm  
    
    def query(self, question: str, context: str = None):
        """
        Query the model with a question, optionally using previous results as context.
        
        :param question: The question to be asked.
        :param context: Optional context to be included in the query.
        :return: The result of the query.
        """
        if not self.chain:
            raise RuntimeError("Chain not set up. Please call setup_chain() before querying.")
        
        # Create the final query text
        final_query = f"{context}\n\n{question}" if context else question
        
        print(f"{self.agent_name} - Query: {final_query}")
        if isinstance(self.chain, Ollama):
            # For direct LLM query without retrieval
            result = self.chain(final_query)
        else:
            # For RetrievalQA with retriever
            result = self.chain.invoke({"query": final_query})
        
        # Store the result of the query
        self.previous_results[question] = result['result'] if isinstance(result, dict) else result
        
        return result['result'] if isinstance(result, dict) else result


# Example usage:

# Initialize RAGTools
print("--------------------------")
print(f"Tool 1")
print("--------------------------")
tool1 = RAGTool(
    name ="tool1",
    file_paths=[
    "/Users/markusfreyt/Development/Projects/AI/langchain/docs/TPLINK1.txt"
    ]
)

print("--------------------------")
print(f"Tool 2")
print("--------------------------")
tool2 = RAGTool(
    name ="tool2",
    file_paths=[
    "/Users/markusfreyt/Development/Projects/AI/langchain/docs/TPLINK2.txt"
    ]
)

# Initialize and set up Agents
agent1 = Agent(name="agent1", model="llama3.1", rag_tool=tool1)
agent2 = Agent(name="agent2", model="llama3.1", rag_tool=tool2)
agent3 = Agent(name="agent3", model="llama3.1")

agent1.setup_chain()
agent2.setup_chain()
agent3.setup_chain()

print("--------------------------")
print(f"Agent 1")
print("--------------------------")
first_query_agent1 = "What is the price of this product and summarize its features?"
first_result_agent1 = agent1.query(first_query_agent1)
print("--------------------------")
print(f"Result: {first_result_agent1}")
print("--------------------------")

print("--------------------------")
print(f"Agent 2")
print("--------------------------")
first_query_agent2 = "What is the price of this product and summarize its features?"
first_result_agent2 = agent2.query(first_query_agent2)
print("--------------------------")
print(f"Result: {first_result_agent2}")
print("--------------------------")

print("--------------------------")
print(f"Agent 3")
print("--------------------------")
summary_query = "Based on the output from Agent1 and Agent2, name the cheaper product and summarize its features."
context = f"Agent 1 Result: {first_result_agent1}\n\nAgent 2 Result: {first_result_agent2}"
summary_result = agent3.query(summary_query, context=context)
print("--------------------------")
print(f"Result: {summary_result}")
print("--------------------------")
