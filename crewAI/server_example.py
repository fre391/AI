import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]
serper_api_key = os.environ["SERPER_API_KEY"]

search_tool = SerperDevTool()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Getting to know where to swim in Natural Water in Bielefeld',
  backstory="""You work as a travel agent in Bielefeld.""",
  verbose=True,
  allow_delegation=False,
  llm = ChatGroq(
    api_key=groq_api_key,
    model="llama3-70b-8192"
  ),
  tools=[search_tool]
)
writer = Agent(
  role='Blog Writer',
  goal='Write short blog posts for travel guides',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.""",
  verbose=True,
  allow_delegation=True,
  llm = ChatGroq(
    api_key=groq_api_key,
    model="llama3-70b-8192"
  ),
)

# Create tasks for your agents
task1 = Task(
  description="""Find ten places where you can swim in Natural Water""",
  expected_output="list in bullet points with ten results",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post in german. Your post should be informative yet accessible with a list of places to swim in Natural Water.""",
  expected_output="Full blog post of at least 4 paragraphs",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2, # You can set it to 1 or 2 to different logging levels
  process = Process.sequential
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)