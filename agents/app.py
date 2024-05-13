from langchain_community.tools import wikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
tool = wikipediaQueryRun(api_wrapper)



