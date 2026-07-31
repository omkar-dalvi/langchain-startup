import asyncio
import os
import ssl 
import certifi 
from typing import List, Any, Dict 
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_pinecone import PineconeVectorStore

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv('../.env')

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small', show_progress_bar=True, chunk_size=50, retry_min_seconds=10)
vectorstore = PineconeVectorStore(index_name='langchain-documentation-2026', embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=100, max_pages=1000)
tavily_crawl = TavilyCrawl()






async def main():
  print("Hello from main()")
  
  log_header("DOCUMENTATION INGESTION PIPELINE")
  
  log_info("Crawling information from Langchain site")
  
  res = tavily_crawl.invoke(
    {
      "url": "https://python.langchain.com",
      "max_depth": 5,
      "extract_depth": "advanced"
    }
  )
  
  all_docs = [Document(page_content=doc['raw_content'], metadata={"source": doc['url']}) for doc in res['results']]
  
  log_success(f"Successfully crawled {len(all_docs)} from Langchain site.")
  
  
  
  
  
if __name__ == "__main__":
  asyncio.run(main())
  pass