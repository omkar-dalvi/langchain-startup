
import os
from dotenv import load_dotenv
from typing import Dict, Any

from langchain.agents import create_agent
from langchain.messages import ToolMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain.chat_models import init_chat_model

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Loading the environment variables
load_dotenv('../../.env')

# Initializing embedding model
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Initializing VectorStore
vectorstore = PineconeVectorStore(index_name='langchain-doc-assistant-2026', embedding=embeddings)

# Initialize Chat Model
model = init_chat_model('gpt-5.4-nano', model_provider='openai')

# Defining the retriever tool

@tool(response_format='content_and_artifact')
def retrieve_context(query: str):
  """Retrieve relevant information to help answer user query related to Langchain

  Args:
      query (str): User query
  """
  
  retrieved_docs = vectorstore.as_retriever().invoke(query, k=3)
  content = "\n\n".join(
    [f"Source: {doc.metadata.get('source')} \n\n Content: {doc.page_content}" for doc in retrieved_docs] 
  )
  
  return content, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
  """Run the RAG pipeline to answer user query

  Args:
      query (str): User query

  Returns:
      Dict[str, Any]: Dictionary containing the response and metadata
  """
  
  # Define the system prompt
  system_prompt = "\n".join([
    "You are a helpful AI assistant that help user answer query about Langchain documentation",
    "You have access to a tool that can help you to get relevant documentation of Langchain",
    "ALWAYS use the tool for retrieving any information related to Langchain",
    "ALWAYS cite the sources in your answers",
    "If you cannot find the answer from the retrieved documents of the tool, do not use any other information to answer and simply say DIDN'T FIND ANY ANSWER IN THE DATABASE"
  ])

  # Creating the Agent
  agent = create_agent(model=model, tools=[retrieve_context], system_prompt=system_prompt)
  
  # Invoke the Agent
  response = agent.invoke({
      "messages": [HumanMessage(content=query)]
    }
  )
  
  # Extract the last message as the response
  answer = response['messages'][-1].content
  
  # Extract artifact
  context_docs = []
  
  for message in response['messages']:
    # Check if it is ToolMessage and has artifact attribute
    if isinstance(message, ToolMessage) and hasattr(message, 'artifact'):
      if isinstance(message.artifact, list):
        context_docs.extend(message.artifact)
  
  return {
    "answer": answer,
    "artifact": context_docs 
  }
  
if __name__ == '__main__':
  result = run_llm(query="What are Deep Agents?")
  print(result)
  
  
  
  
  




