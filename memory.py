from dotenv import load_dotenv
load_dotenv()
import argparse
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory

# ConversationBufferMemory
def get_conversation_chain(memory_type, **memory_type_kwargs):
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
    memory_class = globals()[memory_type]
    memory = memory_class(**memory_type_kwargs)
    print("Memory Variables:", memory.load_memory_variables())
    # Save context using memory.save_context
    conversation = ConversationChain(
        llm=llm, 
        memory=memory,
        verbose=True
    )
    return conversation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=str, default="ConversationBufferMemory")
    parser.add_argument("--window_size", type=int, default=10)
    args = parser.parse_args()
    memory_type = args.memory
    memory_kwargs = {}
    print(memory_type)
    if memory_type == "ConversationBufferWindowMemory":
        memory_kwargs["k"] = args.window_size
    if memory_type in ["ConversationTokenBufferMemory", "ConversationSummaryBufferMemory"]:
        memory_kwargs["max_token_limit"] = args.token

    print(memory_kwargs)
    conversation = get_conversation_chain(memory_type, **memory_kwargs)
    while True:
        text = input("You: ")
        print(conversation.predict(input=text))
        if text.lower() == "bye":
            break
