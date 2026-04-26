from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


load_dotenv()
def main():
    information = """
    Elon musk is the CEO of SpaceX and Tesla. His hobbies include space exploration, electric vehicles, and artificial intelligence. He is also known for his philanthropic efforts and has donated to various causes.
    """
    
    summary_template = """
    Given the information about the person, give me the following:
    1. 10 word summary
    2. Two hobbies
    
    Information
    {information}
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    
    llm = ChatOpenAI(temperature=0, model="gpt-5.4-nano")
    # llm = ChatOllama(temperature=0, model='gemma3:270m')
    
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)
    
    
    
    
    
    
if __name__ == "__main__":
    main()
