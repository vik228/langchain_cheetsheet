import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("data/attention.pdf")
pages = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(pages)
db = FAISS.from_documents(texts, embedding=OpenAIEmbeddings())

prompt = ChatPromptTemplate.from_template("""
Answer the following question using the context provided.
Think step by step before providing a detailed answer.
<context>
{context}
</context>
Question: {input}                      
""")

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
document_chain = create_stuff_documents_chain(llm, prompt)

retriever = db.as_retriever()
retrival_chain = create_retrieval_chain(retriever, document_chain)

result = retrival_chain.invoke({"input": "Scaled dot product attention explain kario?"})
print(result['answer'])








