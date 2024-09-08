from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
from session_manager import SessionManager

session_manager = SessionManager("session_history.json")

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

model = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=groq_api_key)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Du bist ein hilfreicher Assistent. Beantworte alle Fragen nach bestem Wissen."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialisieren des SessionManagers
session_manager = SessionManager("session_history.json")

def response_with_template(session_id, message):
    try:
        # Nachrichten zur Historie hinzufügen
        history = session_manager.get_history(session_id)
        message_dict = {"type": "human", "content": message}
        session_manager.add_message(session_id, message_dict)
        
        # Formatieren des Prompts
        formatted_prompt = prompt.invoke({"messages": [session_manager.dict_to_message(m) for m in history]})
        formatted_messages = formatted_prompt.messages
        
        # Streaming der Antwort
        response_stream = model.stream(formatted_messages)
        response_content = ""
        for r in response_stream:
            response_content += r.content
            print(r.content, end="", flush=True)
        
        # Hinzufügen der Antwort zur Historie
        ai_message_dict = {"type": "ai", "content": response_content}
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
