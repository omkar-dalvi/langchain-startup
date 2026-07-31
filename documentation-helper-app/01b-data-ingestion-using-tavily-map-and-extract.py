import asyncio
from logger import log_header, log_info, log_error, log_success, log_warning
from langchain_tavily import TavilyMap, TavilyExtract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv('../.env')

tavily_extract = TavilyExtract()

embeddings = OpenAIEmbeddings(
  model='text-embedding-3-small',
  show_progress_bar=True,
  chunk_size=50,
  retry_min_seconds=5
)

vector_store = PineconeVectorStore(index_name="langchain-doc-assistant-2026", embedding=embeddings)


def chunk_urls(urls: List[str], chunk_size=20) -> List[List[str]]:
  """Splits URLs into specified size

  Args:
      urls (List[str]): List of URLs
      chunk_size (int, optional): Chunk size of each batch. Defaults to 20.

  Returns:
      List[List[str]]: List of list of batches
  """
  chunks = []
  for i in range(0, len(urls), chunk_size):
    chunk = urls[i:i+chunk_size]
    chunks.append(chunk)
  return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
  """Extract documents from a batch of URLs

  Args:
      urls (List[str]): List of URLs
      batch_num (int): Batch number

  Returns:
      List[Dict[str, Any]]: Documents extracted from Tavily Extract API
  """
  try:
    log_info(f"[TAVILY EXTRACT] Processing batch {batch_num}")
    docs = await tavily_extract.ainvoke(input={"urls": urls})
    log_success(f"[TAVILY EXTRACT] Completed batch {batch_num}, extracted {len(docs['results'])} documents")
    return docs
  except Exception as e:
    log_error(f"TavilyExtract: Failed to fetch document for batch {batch_num} - {e}")
    
async def async_extract(url_batches: List[List[str]]):
  log_header("DOCUMENT EXTRACTION PHASE")
  
  # Creating list of coroutine objects
  tasks = [extract_batch(batch, batch_num+1) for batch_num, batch in enumerate(url_batches)]
  
  # Concurrently running all coroutine objects using asyncio.gather()
  results = await asyncio.gather(*tasks, return_exceptions=True)
  
  all_pages = []
  failed_batches = 0
  
  for result in results:
    if isinstance(result, Exception):
      log_error(f"TavilyExtract: Batch failed with exception - {result}")
      failed_batches += 1
    else:
      for extracted_page in result['results']:
        document = Document(
          page_content=extracted_page['raw_content'],
          metadata={"source": extracted_page['url']}
        )
        all_pages.append(document)
  
  log_success(f"[TAVILY EXTRACT] Successfully extracted {len(all_pages)} documents")
  
  if failed_batches > 0:
    log_warning(f"[TAVILY EXTRACT] {failed_batches} number of batches failed during extraction")
  
  return all_pages

async def index_documents_async(documents: List[Document], batch_size: int=50):
  """Index documents in Batch

  Args:
      documents (List[Document]): List of documents to be indexed
      batch_size (int, optional): Batch size. Defaults to 50.
  """
  
  log_header("DATA INDEXING")
  log_info(f"[INDEXING] Processing {len(documents)} for indexing")
  
  batches = [documents[i: i+batch_size] for i in range(0, len(documents), batch_size)]
  
  log_info(f"[INDEXING] Splitted into {len(batches)} batches with {batch_size} in each batch")
  
  # Process batch asynchronously
  async def add_batch(batch: List[Document], batch_num:int):
    try:
      await vector_store.aadd_documents(batch)
      log_success(f"[INDEXING] Successfully added batch - {batch_num}")
    except Exception as e:
      log_error(f"[INDEXING] Error in indexing batch {batch_num}: {e}")
      return False
    return True
  
  tasks = [add_batch(batch, batch_num+1) for batch_num, batch in enumerate(batches)]
  results = await asyncio.gather(*tasks, return_exceptions=True)
  
  successful = sum(1 for result in results if result is True)
  
  if successful == len(batches):
    log_success(f"[INDEXING] All batch indexed successfully")
  else:
    log_warning(f"[INDEXING] Processed {successful}/{len(batches)} successfully")

async def main():
  print("Hello from main()")
  
  log_header("DATA INGESTION PIPELINE")
  log_info("Creating site map of https://python.langchain.com using Tavily Map")
  
  tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
  
  site_map = tavily_map.invoke("https://python.langchain.com")
  log_info(f"[TAVILY MAP] Successfully mapped {len(site_map['results'])} URLs from https://python.langchain.com")
  
  url_batches = chunk_urls(site_map['results'], chunk_size=20)
  log_info(f"[URL BATCHING] Splitted the site map into {len(url_batches)} batches")
  
  all_docs = await async_extract(url_batches)
  
  # Splitting the documents into chunks
  log_header("DOCUMENT CHUNKING PHASE")
  log_info(f"[DOCUMENT CHUNKING] Processing {len(all_docs)} documents")
  
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
  splitted_docs = text_splitter.split_documents(all_docs)
  
  log_success(f"[DOCUMENT CHUNKING] Created {len(splitted_docs)} chunks from {len(all_docs)} documents  ")
  
  await index_documents_async(splitted_docs, batch_size=500)
  
  
  
  

if __name__ == '__main__':
  asyncio.run(main())