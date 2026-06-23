
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
import os

load_dotenv()

def run_rag():
  print(f"Ingesting data...")
  loader = TextLoader("./docs/medium.txt", autodetect_encoding=True)
  document = loader.load()
  
  print(f"Splitting data...")
  print(f"Splitting data into chunks")
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_documents(document)
  print(f"Created {len(texts)} chunks")
  
  embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))
  
  print(f"Storing in vector database")
  PineconeVectorStore.from_documents(documents=texts, embedding=embeddings, index_name=os.environ.get('INDEX_NAME'))
  
    
  

if __name__ == "__main__":
  run_rag()