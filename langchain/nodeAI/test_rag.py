import os
import warnings
import json

from utils import Agent, RAGTool

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

print()
print()
print("--------------------------")
print(f"Tool 1")
print("--------------------------")
tool1 = RAGTool(
    name ="tool1",
    file_paths=[
    "/Users/markusfreyt/Development/Projects/AI/langchain/docs/TPLINK1.txt"
    ]
)

print()
print("--------------------------")
print(f"Tool 2")
print("--------------------------")
tool2 = RAGTool(
    name ="tool2",
    file_paths=[
    "/Users/markusfreyt/Development/Projects/AI/langchain/docs/TPLINK2.txt"
    ]
)

# Initialize and set up Agents
agent1 = Agent(name="agent1", provider="Ollama", model="llama3.1", tool=tool1)
agent2 = Agent(name="agent2", provider="Ollama", model="llama3.1", tool=tool2)
agent3 = Agent(name="agent3", provider="ChatGroq", model="llama3-8b-8192", api_key=groq_api_key)

print()
print()
print("--------------------------")
print(f"Agent 1")
print("--------------------------")
agent1_query = "Wie teuer ist das Produkt und welche Features hat es?"
print(f"Query: {agent1_query}")
agent1_result = agent1.query(agent1_query)
print(f"Result: {agent1_result}")
print("--------------------------")
print(f"Status: {print(json.dumps(agent1.get_status(), indent=4))}")
print("--------------------------")


print()
print()
print("--------------------------")
print(f"Agent 2")
print("--------------------------")
agent2_query = "What is the price of this product and summarize its features?"
print(f"Query: {agent2_query}")
agent2_result = agent2.query(agent2_query)
print(f"Result: {agent2_result}")
print("--------------------------")
print(f"Status: {print(json.dumps(agent2.get_status(), indent=4))}")
print("--------------------------")


print()
print()
print("--------------------------")
print(f"Agent 3")
print("--------------------------")
agent3_query = "Based on the output from Agent1 and Agent2, name the cheaper product and summarize its features."
print(f"Query: {agent3_query}")
context = f"Agent 1 Result: {agent1_result}\n\nAgent 2 Result: {agent2_result}"
agent3_result = agent3.query(agent3_query, context)
print(f"Result: {agent3_result}")
print("--------------------------")
print(f"Status: {print(json.dumps(agent3.get_status(), indent=4))}")
print("--------------------------")