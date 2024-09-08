from crewai import Agent, Task, Crew, Process 
import os 

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]
openai_model_name = os.environ["OPENAI_MODEL_NAME"]
openai_api_base = os.environ["OPENAI_API_BASE"]
serper_api_key = os.environ["SERPER_API_KEY"]

# Code snippet to be explained 
codeSnippet = '''
  print("Hello, World!")
'''

# Define the explainer agent 
explainer = Agent( 
  role="code explainer", 
  goal="Take in a code snippet and explain how it works and what it does in natural language", 
  backstory="You are a Senior Software engineer at a tech company and are tasked with explaining a code snippet to a blog writer who is writing a blog post on the codebase.",
  verbose=True,
  allow_delegation=False 
)

# Define the blog writer agent 
blogWriter = Agent( 
  role="blog writer", 
  goal="Take an explanation of a code snippet provided by the code explainer agent and write an explanatory blog post for children in german on it. Summarize in just one sentence.",

  backstory="You are a blog writer for kids who is writing a blog post on a codebase. You need to understand the code explanation provided by the code explainer agent and write a blog post for kids in german on it.", 
  verbose=True, 
  allow_delegation=False 
) 

# Task to explain the code snippet 
explainSnippet = Task( 
  description=f"Explain '{codeSnippet}' a code snippet", 
  agent=explainer, 
  expected_output="An explanation of the code snippet in natural language" 
) 

# Task to write a blog post based on the explanation 
writeBlogPost = Task( 
  description=f"Write a blog post on this code snippet: '{codeSnippet}', using the explanation provided by the code explainer agent. Include relevant sections of the code with your explanations.", 
  agent=blogWriter, 
  expected_output="A blog post containing code and explanations of the code. The explanation should be in natural language for Kids using 'Du' and should be based on the explanation provided by the previous agent for the code" 
) 

# Define the crew and process 
crew = Crew( 
  agents=[explainer, blogWriter],
  tasks=[explainSnippet, writeBlogPost], 
  verbose=1, 
  process=Process.sequential )

# Execute the process and print the output 
output = crew.kickoff() 
print("-------------------------------------")
print(output)        