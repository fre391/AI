import os
from typing import Dict, Union, Any, List
import langchain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
langchain_api_key = os.environ["LANGCHAIN_API_KEY"]
groq_api_key = os.environ["GROQ_API_KEY"]

# Custom StdOutCallbackHandler to control chain execution output
class CustomStdOutCallbackHandler(StdOutCallbackHandler):
    def __init__(self, event_out = False, details_out = False):
        self.event_out = event_out
        self.details_out = details_out
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        if self.event_out: print("\033[96m LLM Started... \033[0m")
        if self.details_out: print(f"Serialized: {serialized}")
        if self.details_out: print(f"Prompts: {prompts}")

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> Any:
        if self.event_out: print("\033[95m Chat Model Started... \033[0m")
        if self.details_out: print(f"Serialized: {serialized}")
        if self.details_out: print(f"Messages: {messages}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        if self.event_out: print(f"\033[94m New Token: {token} \033[0m")

    def on_llm_end(self, response: Dict[str, Any], **kwargs: Any) -> Any:
        if self.event_out: print("\033[92m LLM Ended... \033[0m")
        if self.details_out: print(f"Response: {response}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        if self.event_out: print("\033[91m LLM Error... \033[0m")
        if self.details_out: print(f"Error: {error}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        if self.event_out: print("\033[93m Chain Started... \033[0m")
        if self.details_out: print(f"Serialized: {serialized}")
        if self.details_out: print(f"Inputs: {inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        if self.event_out: print("\033[92m Chain Ended... \033[0m")
        if self.details_out: print(f"Outputs: {outputs}")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        if self.event_out: print("\033[91m Chain Error... \033[0m")
        if self.details_out: print(f"Error: {error}")

custom_handler = CustomStdOutCallbackHandler(event_out = False, details_out = False)
langchain.debug = False

prompt = ChatPromptTemplate.from_messages([
    ('system', "Translate the following into {language} and transform to JSON Format:"),
    ('user', '{text}')
])
model = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=groq_api_key)
parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"text": "Habe spanisch in der Schule gelernt, aber vieles vergessen?", "language": "spanish"}, {"callbacks": [custom_handler]})
print("---------------------------------------------------------")
print(f"Final Response: {response}")
print("---------------------------------------------------------")
