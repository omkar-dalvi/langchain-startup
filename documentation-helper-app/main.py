from typing import Any, Dict, List
import streamlit as st
from dotenv import load_dotenv

load_dotenv("../.env")

from backend.core import run_llm


def format_sources(context_docs: List[Any]) -> List[str]:
    """Returns the list of sources from context docs

    Args:
        context_docs (List[Any]): List of Documents

    Returns:
        List[str]: List of sources
    """
    return [
        str((meta.get("source") or "Unknown"))
        for doc in context_docs
        if (meta := getattr(doc, "metadata"), None) is not None
    ]


st.set_page_config(page_title="Langchain Document Helper", layout="centered")
st.title("Langchain documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
  st.session_state.messages = [
    {
    "role": "assistant",
    "content": "Ask me anything about Langchain. I can give your answers along with its citations",
    "sources": []
    }
  ]

for message in st.session_state.messages:
  with st.chat_message(message['role']):
    st.markdown(message['content'])
    if message.get('sources', None):
      with st.expander("Sources"):
        for source in message.get('sources'):
          st.markdown(f"- {source}")


prompt = st.chat_input("Ask a question about Langchain")

if prompt:
  st.session_state.messages.append({
    "role": "user",
    "content": prompt,
    "sources": []
  })
  with st.chat_message("user"):
    st.markdown(prompt)
  
  with st.chat_message("assistant"):
    try:
      with st.spinner("Retrieving docs and generating answer..."):
        result = run_llm(prompt)
        answer = str(result.get('answer', '')).strip() or "No answer generated"
        sources = format_sources(result.get('artifact', []))
      st.markdown(answer)
      
      if sources:
        with st.expander("Sources"):
          for s in sources:
            st.markdown(f"-{s}")
      
      st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
      })
      
      
    except Exception as e:
      st.error("Failed to generate a response")
      st.exception(e)
      

  
  