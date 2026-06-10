from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3.5:0.8b"

# DEFINING TOOLS


# Tool 1
@tool
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
@tool
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
    return round(price * (1 - discount / 100), 2)


@traceable
def run_agent(query: str):
    # Defining tool lists
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    # Defining llms and adding tools
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"User query: {query}")
    print("*" * 60)

    # Defining messages list
    messages = [
        SystemMessage(content=""" 
      You are a helpful shopping assistant.
      You have access to a product catalog tool and a discount tool.
      STRICT RULES - You must follow these exactly.
      1. NEVER guess or assume the prices of the product. You MUST call the get_product_price tool first to get the real price
      
      2. Only call apply_discount AFTER you have received price of the product from get_product_price tool. Pass the exact price returned by the get_product_price tool. Do NOT pass madeup numbers
      
      3. Never calculate discounts yourself using math. Always use the apply_discount tool for calculating discount
      
      4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume the tier
      
      5. The name of the product will be in the user query. It will be one of the following {laptop, mobile, keyboard}. If you see any one of these, then it is the product item for which price needs to be fetched, discount should be applied and final discounted price should be returned
      
      
      """),
        HumanMessage(content=query),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n-- Iteration {iteration} --")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool call, then it is the final answer
        if not tool_calls:
            print(f"\n Final answer: {ai_message.content}")
            return ai_message.content

        # Getting the first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_id = tool_call.get("id")

        print(f"\n [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"\n [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_id))
    
    print("ERROR: Max iterations reached without final answer")
    return None

    pass


if __name__ == "__main__":
    result = run_agent(
        query="What is the price of the laptop after applying gold tier discount?"
    )
    print(result)
    pass
