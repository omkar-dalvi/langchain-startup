from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from typing import List 
from pydantic import BaseModel, Field 

class Source(BaseModel):
  """Schema for a source used by the agent"""
  url: str = Field(description="The URL of the source")
  
class AgentResponse(BaseModel):
  """Schema for agent response with answer and sources"""
  answer: str = Field(description="The agent's answer of the query")
  sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")
  
load_dotenv()



def main():
    llm = ChatOpenAI(temperature=0, model='gpt-5.4-nano')
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
    response = agent.invoke({'messages': [HumanMessage(content="What the whether in Tokyo?")]})
    
    # OR
    # response = llm.with_structured_output(AgentResponse)
    # print(response)
    
    
    
    
if __name__ == "__main__":
    main()
