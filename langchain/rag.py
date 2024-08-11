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

    def query(self, question: str):
        """
        Query the loaded documents with a question.
        
        :param question: The question to be asked.
        :return: The result of the query.
        """
        if not self.chain:
            raise RuntimeError("No documents loaded. Please load documents before querying.")
        
        print(f"Agent query: {question}")
        result = self.chain.invoke({"query": question})
        return result['result']

# Example usage:

pdf_paths = [
    "/Users/markusfreyt/Development/Projects/AI/langchain/docs/test.pdf",
    # Add more PDF file paths as needed
]

txt_paths = [
    "/Users/markusfreyt/Development/Projects/AI/langchain/docs/kunst.txt",
    "/Users/markusfreyt/Development/Projects/AI/langchain/docs/test.txt",
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
result = rag.query("In the documents information about three topics Covid, Burg Eltz and Kunst are provided. Can you please summarize each of all three topics by only one sentence in german?")
print("--------------------------")
print(result)
print("--------------------------")

result = rag.query("Wo liegt Burg Eltz?")
print("--------------------------")
print(result)
print("--------------------------")

result = rag.query("Was war die ursprüngliche Bedeutung des Begriffs Kunst ?")
print("--------------------------")
print(result)
print("--------------------------")

result = rag.query("Wie lange muss man nach Ansteckung mit Covid eine Maske tragen?")
print("--------------------------")
print(result)
print("--------------------------")




#-----------------------------------------
# Expected Output:
#-----------------------------------------

"""
Loading PDF: /Users/markusfreyt/Development/Projects/AI/langchain/docs/test.pdf...
   Splitting into chunks...
   Number of chunks created : 3
     -->  Mein PCR-TEST ist positiv– was muss ich jetzt tun?
     -->  Sie regelmäßig alle Zimmer der Wohnung. Wenn Sie k
     -->  4. Informieren Sie Ihre Haushaltsangehörigen!• Tei
   Adding to the collection...
   Creating embeddings for document chunks...
   Number of chunks in FAISS index: 3
   Setting up the retrieval chain...

Loading TXT: /Users/markusfreyt/Development/Projects/AI/langchain/docs/kunst.txt...
   Splitting into chunks...
   Number of chunks created: 2
     -->  Das Wort Kunst (lateinisch ars, griechisch téchne[
     -->  Literatur mit den Hauptgattungen Epik, Dramatik, L
   Adding to the collection...
   Creating embeddings for document chunks...
   Number of chunks in FAISS index: 5
   Setting up the retrieval chain...

Loading TXT: /Users/markusfreyt/Development/Projects/AI/langchain/docs/test.txt...
   Splitting into chunks...
   Number of chunks created: 2
     -->  Die Burg Eltz ist das Ergbnis eines kreativen Proz
     -->  den Eltz-Kempenich, genannt „Eltz vom goldenen Löw
   Adding to the collection...
   Creating embeddings for document chunks...
   Number of chunks in FAISS index: 7
   Setting up the retrieval chain...

Agent query: In the documents information about three topics Covid, Burg Eltz and Kunst are provided. Can you please summarize each of all three topics by only one sentence in german?
--------------------------
Hier sind die Zusammenfassungen:

1. **Covid**: (Es gibt keine Informationen zum Thema Covid in den gegebenen Dokumenten.)
2. **Burg Eltz**: Die Burg Eltz ist eine Höhenburg aus dem 12. Jahrhundert und eine der bekanntesten ihrer Art in Deutschland.
3. **Kunst**: Kunst ist ein menschliches Kulturprodukt, das Ergebnis eines kreativen Prozesses, und umfasst verschiedene Ausdrucksformen wie bildende Kunst, Musik und Literatur.
--------------------------
Agent query: Wo liegt Burg Eltz?
--------------------------
Die Burg Eltz liegt im Tal der Elz, einem linken Nebenfluss der Mosel, in der Gemarkung der Ortsgemeinde Wierschem in Rheinland-Pfalz. Sie befindet sich auf 129 m ü. NHN.
--------------------------
Agent query: Was war die ursprüngliche Bedeutung des Begriffs Kunst ?
--------------------------
Die ursprüngliche Bedeutung des Begriffs Kunst wurde auf alle Produkte menschlicher Arbeit angewandt (vgl. Kunstfertigkeit) als Gegensatz zur Natur, was beispielsweise bei Kunststoff, Künstlicher Ernährung, Künstlichem Aroma, Künstlicher Intelligenz ersichtlich wird.
--------------------------
Agent query: Wie lange muss man nach Ansteckung mit Covid eine Maske tragen?
--------------------------
Nach dem gegebenen Text ist die Antwort auf Ihre Frage 5 Tage. Die Verpflichtung zum Tragen einer Maske oder zur Absonderung endet nämlich 5 Tage nach Erstnachweis des Erregers (Datum der Probeentnahme oder Laboreingangsdatum, je nachdem was auf dem Nachweis steht).
--------------------------
"""
