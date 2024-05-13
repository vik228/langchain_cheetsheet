from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


def setup_chain():
    llm = ChatOpenAI(
        temperature=0.9,
        model="gpt-3.5-turbo"
    )
    prompt = ChatPromptTemplate.from_template(
        "what is the best name to describe an\
        animal with the following {description}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

if __name__ == "__main__":
    chain = setup_chain()
    while True:
        description = input("Enter a description of an animal: ")
        print(chain.run(description))
        if description == "bye":
            break



