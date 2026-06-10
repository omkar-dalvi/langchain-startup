from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

load_dotenv()

# Create Tavily client
tavily = TavilyClient()

# Define Langchain tool for searching on Net

@tool
def search(query: str) -> str:
  """Tool that searches over the internet

  Args:
      query (str): The query to search for

  Returns:
      str: The search result
  """
  print(f"Searching for {query}")
  return tavily.search(query=query)



def main():
    llm = ChatOpenAI(temperature=0, model='gpt-5.4-nano')
    tools = [search]
    agent = create_agent(model=llm, tools=tools)
    response = agent.invoke({'messages': [HumanMessage(content="What the whether in Tokyo?")]})
    print(response)
    
    
    
    
if __name__ == "__main__":
    main()
