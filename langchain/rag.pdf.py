import warnings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader

warnings.filterwarnings('ignore')

print("Load Data...")
loader = PDFPlumberLoader("/Users/markusfreyt/Development/Projects/AI/langchain/docs/Blockchain.pdf")
docs = loader.load()

print("Split the document into chunks...")
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(docs)

print("Create embeddings for each text chunk​...")
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(texts, embeddings)

print("Retrieval from the vector database...")
llm = Ollama(model="llama3.1")
chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever()
)
question = "Schreibe eine Zusammenfassung"
result = chain.invoke({"query": question})

print("Output Result...")
print(result['result'])