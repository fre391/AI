import langchain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from session_manager import SessionManager
from custom_stdout_callback_handler import CustomStdOutCallbackHandler

# Initialize SessionManager
session_manager = SessionManager("session_history.json")

# Define CustomStdOutCallbackHandler
custom_handler = CustomStdOutCallbackHandler(event_out = True, details_out = True)
langchain.debug = False

# Load environment variables
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

# Initialize the ChatGroq model
model = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=groq_api_key)

# Create the ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Du bist ein hilfreicher Assistent. Beantworte alle Fragen nach bestem Wissen."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize StrOutputParser
parser = StrOutputParser()

# Create the chain
chain = prompt | model | parser

def response_with_template(session_id, message):
    try:
        # Add message to history
        history = session_manager.get_history(session_id)
        message_dict = {"type": "human", "content": message}
        session_manager.add_message(session_id, message_dict)
        
        # Prepare the messages for the prompt
        formatted_messages = [session_manager.dict_to_message(m) for m in history]
        
        # Create the input for the chain
        chain_input = {"messages": formatted_messages}
        callback = {"callbacks": [custom_handler]}
        
        # Invoke the chain
        response = chain.invoke(chain_input, callback)
        
        # Print response
        print(response, end="", flush=True)
        
        # Add response to history
        ai_message_dict = {"type": "ai", "content": response}
        session_manager.add_message(session_id, ai_message_dict)
        
        print("\n---")
    except Exception as e:
        print(f"Fehler: {e}")

if __name__ == "__main__":
    print("Geben Sie eine Session-ID ein oder 'exit', um das Programm zu beenden.")
    session_id = input("Session-ID: ")
    
    while True:
        user_input = input("\n\nDu: ")
        if user_input.lower() == "exit":
            print("Auf Wiedersehen!")
            break
        print("Assistent: ", end="")
        response_with_template(session_id, user_input)
        print("\n---")
