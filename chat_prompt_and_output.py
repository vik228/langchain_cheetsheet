"""
    ChatPromptTemplate and Output parser
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema


def get_completion(message, prompt_template, format_instructions, **kwargs):
    llm_model = kwargs.get("llm_model", "gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template
    )
    messages = prompt.format_messages(
        text=message,
        format_instructions=format_instructions
    )
    chat = ChatOpenAI(
        temperature=0.0,
        model=llm_model
    )
    return chat(messages)

def get_response_schema():
    gift_schema = ResponseSchema(
        name="gift",
        description="Was the item purchased\
                    as a gift for someone else? \
                    Answer True if yes,\
                    False if not or unknown.")
    delivery_days_schema = ResponseSchema(
        name="delivery_days",
        description="How many days\
                    did it take for the product\
                    to arrive? If this \
                    information is not found,\
                    output -1.")
    price_value_schema = ResponseSchema(
        name="price_value",
        description="Extract any\
                    sentences about the value or \
                    price, and output them as a \
                    comma separated Python list.")
    return [
        gift_schema,
        delivery_days_schema,
        price_value_schema
    ]

    return gift_schema

if __name__ == "__main__":
    customer_review = """\
        This leaf blower is pretty amazing.  It has four settings:\
        candle blower, gentle breeze, windy city, and tornado. \
        It arrived in two days, just in time for my wife's \
        anniversary present. \
        I think my wife liked it so much she was speechless. \
        So far I've been the only one using it, and I've been \
        using it every other morning to clear the leaves on our lawn. \
        It's slightly more expensive than the other leaf blowers \
        out there, but I think it's worth it for the extra features.
    """
    review_template_2 = """\
        For the following text, extract the following information:

        gift: Was the item purchased as a gift for someone else? \
        Answer True if yes, False if not or unknown.

        delivery_days: How many days did it take for the product\
        to arrive? If this information is not found, output -1.

        price_value: Extract any sentences about the value or price,\
        and output them as a comma separated Python list.

        text: {text}

        {format_instructions}
    """
    response_schema = get_response_schema()
    output_parser = StructuredOutputParser.from_response_schemas(response_schema)
    format_instruction = output_parser.get_format_instructions()
    output = get_completion(
        customer_review,
        review_template_2,
        format_instruction
    )
    output_dict = output_parser.parse(output.content)
    print(output_dict)


