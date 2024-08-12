   
import os
import warnings
import json

from utils import Agent, RAGTool, WebTool

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

web_tool = WebTool(name="ExampleWebTool", urls=[
    'https://de.wikipedia.org/wiki/Burg_Eltz',
    'https://de.wikipedia.org/wiki/Burg_Steen'
])

agent1 = Agent(name="agent1", provider="Ollama", model="llama3.1", rag_tool=web_tool)
print()
print()
print("--------------------------")
print(f"Agent 1")
print("--------------------------")
agent1_query = "Bei welchen Städten liegen die Burgen?"
print(f"Query: {agent1_query}")
agent1_result = agent1.query(agent1_query)
print(f"Result: {agent1_result}")
print("--------------------------")
print(f"Status: {print(json.dumps(agent1.get_status(), indent=4))}")
print("--------------------------")


