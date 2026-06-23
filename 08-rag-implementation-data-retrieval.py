import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from operator import itemgetter

load_dotenv()

print("Initializing components")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model='gpt-5.4-nano')

vector_store = PineconeVectorStore(
  embedding=embeddings,
  index_name=os.environ.get('INDEX_NAME')
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = PromptTemplate.from_template(
  """
  Answer the question based only on the given context
  
  Context: {context}
  Question: {question}
  
  Provide a detailed answer:
  """
)

def format_docs(docs):
  """Format the documents into a single string

  Args:
      docs (list): List of documents
  """
  return "\n\n".join(doc.page_content for doc in docs)

def create_retrieval_chain_with_lcel():
  """Create a retrieval chain using LCEL (LangChain Expression Language)
  """
  retrieval_chain = (
    RunnablePassthrough.assign(
      context=itemgetter('question') | retriever | format_docs
    )
    | prompt_template
    | llm 
    | StrOutputParser()
  )
  return retrieval_chain
  

if __name__ == "__main__":
  print("Retrieval")
  query = "What is Pinecone in machine learning?"
  
  print("Implementation using LCEL")
  chat_with_lcel = create_retrieval_chain_with_lcel()
  result = chat_with_lcel.invoke({"question": query})
  print(f"Answer: {result}")