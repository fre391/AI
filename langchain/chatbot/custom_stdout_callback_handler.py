from typing import Dict, Union, Any, List
import langchain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

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
