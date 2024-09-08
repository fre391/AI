import json
import os

class SessionManager:
    def __init__(self, filename):
        self.directory = os.path.dirname(os.path.abspath(__file__))
        self.filename = os.path.join(self.directory, filename)
        self.load_sessions()
    
    def load_sessions(self):
        try:
            with open(self.filename, 'r') as f:
                self.sessions = json.load(f)
        except FileNotFoundError:
            self.sessions = {}
    
    def save_sessions(self):
        with open(self.filename, 'w') as f:
            json.dump(self.sessions, f, indent=4)
    
    def get_history(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]
    
    def add_message(self, session_id, message_dict):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(message_dict)
        self.save_sessions()
    
    def message_to_dict(self, message):
        """Konvertiert eine Nachricht in ein Dictionary."""
        return {
            "type": message.get("type"),
            "content": message.get("content")
        }

    def dict_to_message(self, message_dict):
        """Konvertiert ein Dictionary in eine Nachricht."""
        message_type = message_dict.get("type")
        content = message_dict.get("content")
        
        if message_type == "human":
            return {"type": "human", "content": content}
        elif message_type == "ai":
            return {"type": "ai", "content": content}
        else:
            raise ValueError(f"Unknown message type: {message_type}")
