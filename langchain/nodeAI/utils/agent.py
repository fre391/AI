import requests
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from .ragtool import RAGTool

class Agent:
    def __init__(self, name: str, provider: str,model: str, api_key: str = None, tool: RAGTool = None):
        """
        Initialize the Agent class with a specific model and optionally with RAGTool for document processing.
        
        :param model: The model name for the language model (e.g., 'Ollama', 'ChatGroq').
        :param api_key: Optional API key for models requiring authentication (e.g., ChatGroq).
        :param rag_tool: An optional RAGTool instance for document processing.
        """
        self.configuration = {}
        self.configuration['agent_name'] = name
        self.configuration['provider'] = provider
        self.configuration['model_name'] = model
        self.configuration['api_key'] = api_key
        self.configuration['tool'] = tool

        self.input = ""
        self.context = ""
        self.output = ""
        self.chain = None
        self.history = {}  # Store results of previous queries

        self.setup_chain()
    
    def get_status(self):
        status = {}
        print(self.input)
        status['input'] = self.input
        status['context'] = self.context
        status['output'] = self.output
        status['history'] = self.history    
        return status

    def setup_chain(self):
        """
        Set up the retrieval chain using the document retriever from RAGTool if provided.
        """
        try:
            if self.configuration['provider'] == 'Ollama':
                llm = Ollama(model=self.configuration['model_name'])
                if self.configuration['tool']:
                    retriever = self.configuration['tool'].get_document_retriever()
                    self.chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
                else:
                    retriever = None
                    self.chain = llm
            elif self.configuration['provider'] == 'ChatGroq':
                #if not self.check_online_access():
                #    raise RuntimeError("ChatGroq is not accessible offline.")
                if not self.configuration['api_key']:
                    raise ValueError("API key is required for ChatGroq.")
                # Initialize ChatGroq specific setup here
                llm = ChatGroq(model=self.configuration['model_name'], api_key=self.configuration['api_key'])

                if self.configuration['tool']:
                    retriever = self.configuration['tool'].get_document_retriever()
                    self.chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
                else:
                    retriever = None
                    self.chain = llm
            else:
                raise ValueError(f"Unsupported model: {self.configuration['provider']}")
        except Exception as e:
            print(f"Error during setup_chain: {e}")
            raise

    def check_online_access(self) -> bool:
        """
        Check if ChatGroq is accessible online.
        :return: True if ChatGroq is accessible, False otherwise.
        """
        try:
            response = requests.get('https://api.chatgroq.com/health')
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def query(self, question: str, context: str = None):
        """
        Query the model with a question, optionally using previous results as context.
        
        :param question: The question to be asked.
        :param context: Optional context to be included in the query.
        :return: The result of the query.
        """

        if not self.chain:
            raise RuntimeError("Chain not set up. Please call setup_chain() before querying.")
        
        self.input = question
        self.context = context

        # Create the final query text
        #final_query = f"{context}\n\n{question}" if context else question
        self.input = f"Context: {self.context}\nQuestion: {self.input}"  if self.context else f"Question: {self.input}"

        try:
            if self.configuration['provider'] == 'Ollama':
                # For direct LLM query without retrieval
                self.output = self.chain(self.input)
            elif self.configuration['provider'] == 'ChatGroq':
                # For ChatGroq specific query
                # Ensure `query` method accepts a string
                self.output = self.chain.invoke(self.input).content  # Adjust according to ChatGroq API
            else:
                # For RetrievalQA with retriever
                self.output = self.chain.invoke(self.input)  # Direct string query for RetrievalQA
            
            # Store the result of the query
            self.history[self.input] = self.output['result'] if isinstance(self.output, dict) else self.output

            return self.output['result'] if isinstance(self.output, dict) else self.output
        
        except Exception as e:
            print(f"Error during query: {e}")
            return None
