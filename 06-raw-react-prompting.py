from langsmith import traceable
from dotenv import load_dotenv
import ollama
import re
import inspect

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# DEFINING TOOLS


# Tool 1
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Looks up for price of the product

    Args:
        product (str): Product name

    Returns:
        float: Price for the given product
    """
    print(f"  >> Executing get_product_price(product={product})")
    prices = {"laptop": 1299, "mobile": 999, "keyboard": 89}

    return prices.get(product, 0)


# Tool 2
@traceable(run_type="tool")
def apply_discount(price: float, tier: str) -> float:
    """Applies discount for the given tier on the price of the product

    Args:
        price (float): Price of the product
        tier (str): Tier of the product. Available values: gold, bronze, silver

    Returns:
        float: Final price of the product after applying the discount for the given tier
    """
    print(f"  >> Exceuting apply_discount(price={price}, tier={tier})")
    discount_percentages = {"gold": 25, "silver": 15, "bronze": 5}
    discount = discount_percentages.get(tier, 0)
    print(f"Discount: {discount}")
    # print(type(discount))
    # print(type(price))
    return round(float(price) * (1 - discount / 100), 2)


@traceable(name="ReAct Prompt Ollama Chat", run_type="llm")
def ollama_chat_traced(messages, options):
    return ollama.chat(model=MODEL, options=options, messages=messages)


def get_tool_description(tool_dict):
    descriptions = []
    for tool_name, tool_function in tool_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "/n".join(descriptions)


@traceable(name="ReAct prompting Ollama Agent Loop")
def run_agent(query: str):
    # Defining tool lists
    tools = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    tool_descriptions = get_tool_description(tools)
    tool_names = ",".join(tools.keys())

    print(f"User query: {query}")
    print("*" * 60)

    react_prompt = f"""
    You are a helpful shopping assistant.
    You have access to a product catalog tool and a discount tool.
    STRICT RULES - You must follow these exactly.
    1. NEVER guess or assume the prices of the product. You MUST call the get_product_price tool first to get the real price
    
    2. Only call apply_discount AFTER you have received price of the product from get_product_price tool. Pass the exact price returned by the get_product_price tool. Do NOT pass madeup numbers
    
    3. NEVER calculate discounts yourself using math. Always use the apply_discount tool for calculating discount. This is MANDATORY step
    
    4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume the tier
    
    5. The name of the product will be in the user query. It will be one of the following [laptop, mobile, keyboard]. Don't ask for specific brand or model number. If the user asks from any of the above give items, then it is the product item for which price needs to be fetched, discount should be applied and final discounted price should be returned
    
    6. Do NOT ask follow up question. Assume user has given all the information he/she knows. This is MANDATORY step
    
    7. If the user passes any other product apart from the given list or provides any other discount tier not defined in the list, stop reasoning and return "INVALID INPUT. Product should be either [laptop, mobile, keyboard] and discount tier should be [gold, silver, bronze]". Do no reason further as the user has provided wrong inputs
    
    Answer the following questions as best you can. You have access to the following tools:

    {tool_descriptions}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action in parameter=argument format. Do NOT use any other format. This is MANDATORY format
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {query}
    Thought:
    """

    prompt = react_prompt.format(query=query)
    # print(f"[Prompt] {prompt}")
    scratchpad = ""
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n-- Iteration {iteration} --")

        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        
        # print(f"[Reponse] {response}")

        output = response.message.content
        print(f"[Output] {output}")
        
        print(f"[Parsing] Looking for final answer in LLM output")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        
        if final_answer_match:
          final_answer = final_answer_match.group(1).strip()
          print(f"[Parsed] Final answer: {final_answer}")
          return final_answer

        print(f"[Parsing] Looking for Action and Action Input in the LLM output")
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)
        
        if not action_match or not action_input_match:
          print(f"Unable to parse Action/Action Input from LLM")
          break
      
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]
        
        print(f"[Tool Executing] {tool_name}({args})...")
        
        if tool_name not in tools:
          observation = f"Error {tool_name} not found. Available tools: {tools.keys()}"
        else:
          observation = str(tools[tool_name](*args))
          
        print(f"[Tool Result] {observation}")
        scratchpad += f"{output}\nObservation:{observation}\nThought:"
        
    print("ERROR: Max iterations reached without final answer")
    return None

    pass


if __name__ == "__main__":
    result = run_agent(
        query="What is the price of the mobile after applying silver tier discount?"
    )
    print(result)
    pass
