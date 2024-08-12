   
import os
import warnings
import json

from utils import Agent, RAGTool, WebTool

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

urls=[
    'https://nadine-foto.de/preise/',
]
tool1 = WebTool(name="tool1", urls=urls, subpages=True, max_depth=1)

agent1 = Agent(name="agent1", provider="Ollama", model="llama3.1", tool=tool1)
print()
print()
print("--------------------------")
print(f"Agent 1")
print("--------------------------")
agent1_query = "Welche DienstLeistungen werden angeboten und zu welchem Preis. Erstelle eine übersichtliche Tabelle mit einer kurzen Headline und dem zugehörigen Preis."
agent1_query = "Was kostet ein JGA-Shooting und was beinhaltet das?"
print(f"Query: {agent1_query}")
agent1_result = agent1.query(agent1_query)
print(f"Result: {agent1_result}")
print("--------------------------")
print(f"Status: {json.dumps(agent1.get_status(), indent=4)}")
print("--------------------------")