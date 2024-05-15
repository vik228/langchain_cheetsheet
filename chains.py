import argparse
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

load_dotenv()

# Define templates
templates = {
    "physics": """You are a very smart physics professor... {input}""",
    "math": """You are a very good mathematician... {input}""",
    "history": """You are a very good historian... {input}""",
    "computer science": """You are a successful computer scientist... {input}"""
}

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
    language model select the model prompt best suited for the input. \
    You will be given the names of the available prompts and a \
    description of what the prompt is best suited for. \
    You may also revise the original input if you think that revising\
    it will ultimately lead to a better response from the language model.

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \ name of the prompt to use or "DEFAULT"
        "next_inputs": string \ a potentially modified version of the original input
    }}}}
    ```

    REMEMBER: "destination" MUST be one of the candidate prompt \
    names specified below OR it can be "DEFAULT" if the input is not\
    well suited for any of the candidate prompts.
    REMEMBER: "next_inputs" can just be the original input \
    if you don't think any modifications are needed.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>"""

def setup_router_chain():
    template_infos = [
        {"name": "physics", "description": "Good for answering questions about physics", "prompt_template": templates["physics"]},
        {"name": "math", "description": "Good for answering math questions", "prompt_template": templates["math"]},
        {"name": "history", "description": "Good for answering history questions", "prompt_template": templates["history"]},
        {"name": "computer science", "description": "Good for answering computer science questions", "prompt_template": templates["computer science"]}
    ]

    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
    dest_chains = {
        info["name"]: LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template(info["prompt_template"])
        )
        for info in template_infos
    }

    destinations_str = "\n".join([f"{info['name']}: {info['description']}" for info in template_infos])
    default_chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_template("{input}"))

    router_prompt = PromptTemplate(
        template=MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str),
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    return MultiPromptChain(
        router_chain=router_chain,
        destination_chains=dest_chains,
        default_chain=default_chain,
        verbose=True
    )

def setup_chain(templates):
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")
    return [
        LLMChain(llm=llm, prompt=ChatPromptTemplate.from_template(template[0]), **template[1])
        for template in templates
    ]

def get_templates(chain_type):
    if chain_type == "SimpleSequentialChain":
        return [
            ("what is the best name to describe an animal with the following {description}", {}),
            ("write a short story about the following {animal_name}", {}),
        ]
    elif chain_type == "SequentialChain":
        return [
            ("Translate the following review to english:\n\n{review}", {'output_key': 'english_review'}),
            ("Please summarize the review in 1 sentence:\n\n{english_review}", {'output_key': 'summary'}),
            ("Which language is the following review:\n\n{english_review}", {'output_key': 'language'}),
        ]
    raise ValueError(f"Unknown chain type: {chain_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain_type', type=str, default='')
    args = parser.parse_args()
    chain_type = args.chain_type

    if chain_type == "RouterChain":
        chain = setup_router_chain()
        while True:
            question = input("Enter a question: ")
            print(chain.run(question))
            if question.lower() == "bye":
                break
    else:
        templates = get_templates(chain_type)
        chains = setup_chain(templates)
        chain_kls = globals()[chain_type]
        kwargs = {'verbose': True}
        if chain_type == "SequentialChain":
            kwargs.update({
                'input_variables': ['review'],
                'output_variables': ['english_review', 'summary', 'language']
            })
        overall_chain = chain_kls(chains=chains, **kwargs)

        if chain_type == "SimpleSequentialChain":
            while True:
                description = input("Enter a description of an animal: ")
                print(overall_chain.run(description))
                if description.lower() == "bye":
                    break
        elif chain_type == "SequentialChain":
            review_df = pd.read_csv("data/Data.csv")
            review = review_df.Review[5]
            print(overall_chain.run(review))
