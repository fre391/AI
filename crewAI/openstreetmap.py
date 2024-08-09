import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from osm_query_tool import OsmQueryTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_model_name = os.environ.get("OPENAI_MODEL_NAME")
openai_api_base = os.environ.get("OPENAI_API_BASE")
serper_api_key = os.environ.get("SERPER_API_KEY")

# Initialize tools
search_tool = OsmQueryTool()

# Define agents
osmExpert = Agent(
    role="Openstreetmap Expert",
    goal="Write queries for Overpass API for places to swim in natural water in Bielefeld",
    backstory="I am an expert in OSM data, including elements like nodes, areas, ways, and relations, as well as tags and variations. I write validated Overpass API queries.",
    verbose=False,
    allow_delegation=False,
    llm=ChatOpenAI(model_name="llama3-70b-8192", temperature=0.5),
)

overpassExpert = Agent(
    role="Overpass Expert",
    goal="Get real-time data from Overpass API.",
    backstory="I specialize in retrieving real-time data from OpenStreetMap using Overpass API queries. I ensure to deliver at least ten results.",
    verbose=False,
    allow_delegation=True,
    llm=ChatOpenAI(model_name="llama3-70b-8192", temperature=0.8),
    tools=[search_tool]
)

blogWriter = Agent(
    role='Blog Writer',
    goal='Write ten short blog posts for travel guides',
    backstory="I am skilled in writing engaging blog posts and providing detailed information on specific topics.",
    verbose=False,
    allow_delegation=True,
    llm=ChatOpenAI(model_name="llama3-70b-8192", temperature=0.5),
)

# Define tasks
task1 = Task(
    description="Write a query for Overpass API to request JSON data",
    expected_output="A single string query for Overpass API to retrieve JSON data with details of ten places, including title, headline, description, and lat/lon, in German.",
    agent=osmExpert,
)

task2 = Task(
    description="Send the query to Overpass API and retrieve real-time data",
    expected_output="A JSON list of ten places with details in German, including title, headline, description, and lat/lon.",
    agent=overpassExpert,
    context=[task1],
)

task3 = Task(
    description="Write a German blog article for each place to swim in natural water in Bielefeld",
    expected_output="An HTML file containing ten German blog articles, each describing a place with detailed information, including title, description, lat/lon",
    agent=blogWriter,
    verbose=True,
    context=[task2],
)

# Create and run the crew
crew = Crew(
    agents=[blogWriter, osmExpert, overpassExpert],  # Order of agents: blogWriter requests osmExpert, who requests overpassExpert
    tasks=[task1, task2, task3], 
    verbose=2,
    process=Process.sequential
)

# Execute the process
try:
    result = crew.kickoff()
    print("######################")
    print(result)
    print("######################")
    print("Writing in file ...")
    

    # Save result to HTML file
    with open('openstreetmap.html', 'w') as f:
        f.write(str(result))
except Exception as e:
    print(f"An error occurred: {e}")
