from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
load_dotenv()



def main():
    llm = ChatOpenAI(temperature=0, model='gpt-5.4-nano')
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools)
    response = agent.invoke({'messages': HumanMessage(content="What the whether in Tokyo?")})
    print(response)
    
    
    
    
if __name__ == "__main__":
    main()
