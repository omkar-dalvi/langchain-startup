from langsmith import traceable
from dotenv import load_dotenv
import ollama

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3.5:0.8b"

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
    print(type(discount))
    print(type(price))
    return round(price * (1 - discount / 100), 2)


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages, tools):
    return ollama.chat(model=MODEL, tools=tools, messages=messages)


@traceable(name="Ollama Agent Loop")
def run_agent(query: str):
    # Defining tool lists
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_product_price",
                "description": "Looks up for price of the product",
                "parameters": {
                    "type": "object",
                    "required": ["product"],
                    "properties": {
                        "product": {"type": "string", "description": "Product name"}
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_discount",
                "description": "Applies discount for the given tier on the price of the product",
                "parameters": {
                    "type": "object",
                    "required": ["price", "tier"],
                    "properties": {
                        "price": {
                            "type": "number",
                            "description": "Price of the product",
                        },
                        "tier": {
                            "type": "string",
                            "description": "Tier of the product. Available values: gold, bronze, silver",
                        },
                    },
                },
            },
        },
    ]
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"User query: {query}")
    print("*" * 60)

    # Defining messages list
    messages = [
        {
            "role": "system",
            "content": """ 
      You are a helpful shopping assistant.
      You have access to a product catalog tool and a discount tool.
      STRICT RULES - You must follow these exactly.
      1. NEVER guess or assume the prices of the product. You MUST call the get_product_price tool first to get the real price
      
      2. Only call apply_discount AFTER you have received price of the product from get_product_price tool. Pass the exact price returned by the get_product_price tool. Do NOT pass madeup numbers
      
      3. Never calculate discounts yourself using math. Always use the apply_discount tool for calculating discount
      
      4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume the tier
      
      5. The name of the product will be in the user query. It will be one of the following {laptop, mobile, keyboard}. Don't ask for specific brand or model number. If the user asks from any of the above give items, then it is the product item for which price needs to be fetched, discount should be applied and final discounted price should be returned
      
      6. Do NOT ask follow up question. Assume user has given all the information he/she knows. This is MANDATORY step
      
      7. If the user passes any other product apart from the given list or provides any other discount tier not defined in the list, stop reasoning and return "INVALID INPUT. Product should be either {laptop, mobile, keyboard} and discount tier should be {gold, silver, bronze}". Do no reason further as the user has provided wrong inputs
      
      
      """,
        },
        {"role": "user", "content": query},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n-- Iteration {iteration} --")

        response = ollama_chat_traced(tools=tools, messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # If no tool call, then it is the final answer
        if not tool_calls:
            print(f"\n Final answer: {ai_message.content}")
            return ai_message.content

        # Getting the first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"\n [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use(**tool_args)

        print(f"\n [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})

    print("ERROR: Max iterations reached without final answer")
    return None

    pass


if __name__ == "__main__":
    result = run_agent(
        query="What is the price of the keyboard after applying gold tier discount?"
    )
    print(result)
    pass
