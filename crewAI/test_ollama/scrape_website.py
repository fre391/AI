

import os 
from libs.folder_manager import FolderManager

from crewai import Agent, Task, Crew, Process 
from langchain_ollama import ChatOllama
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv
load_dotenv()
serper_api_key = os.environ["SERPER_API_KEY"]
google_api_key = os.environ["GOOGLEAI_API_KEY"]

fm = FolderManager()
fm.delete_database()

scrapeWebsiteTool = ScrapeWebsiteTool(
    website_url='https://piajennert.de/preise',
    config=dict(
        llm=dict(
            provider="ollama", 
            config=dict(
                model="llama3.1:latest",
            ),
        ),
        embedder=dict(
            provider="ollama", 
            config=dict(
                model="mxbai-embed-large:latest",
            ),
        ),
    )
)

question = input("Please enter your search term: ").strip()

researcher = Agent(
    role="Research Agent",
    goal="get the newest infromation from the given website and search through its content to find relevant answers",
    backstory=(
        """
        The research agent is adept at searching and 
        extracting data from documents, ensuring accurate and prompt responses.
        """
    ),
    verbose=True,
    allow_delegation=False,
    tools=[scrapeWebsiteTool],
    llm=ChatOllama(model="llama3.1:latest", base_url="http://localhost:11434"),
)

writer = Agent(
    role="Professional Writer",
    goal="Write professional text based on the research agent's findings",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The professional writer agent has excellent writing skills and is able to craft 
        clear and concise texts based on the provided information.
        """
    ),
    tools=[scrapeWebsiteTool],
    llm=ChatOllama(model="llama3.1:latest", base_url="http://localhost:11434"),
)

research_task = Task(
    description="Search in the given document(s)",
    agent=researcher,
    expected_output=f"extract all information related to the user's question '{question}' in the language the researcher provided",
    params={"search_query": question},  
)

writer_task = Task(
    description="Write a short and clear answer in the language of the question '{question}'",
    agent=writer,
    expected_output=f"Summarize a short answer to the user's question '{question}' in the language the writer provided",
    params={"search_query": question},  
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writer_task],
    verbose=0,
    process=Process.sequential
)

try:
    result = crew.kickoff()
    print("-------------------------------------")
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")  