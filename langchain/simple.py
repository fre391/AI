import os
import langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False
chat_model = ChatGroq(temperature=0,model_name="llama3-8b-8192",api_key=groq_api_key)
test = chat_model.invoke("Wieviel ist die Wurzel aus 8 in dezimal?")
print (test.content)