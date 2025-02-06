from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
load_dotenv()

# PDF loaders
def pdf_loader():
    loader = PyPDFLoader("data/my_resume.pdf")
    pages = loader.load()
    page = pages[0]
    print(page.page_content[:500])
    print(page.metadata)

#youtube loader
def youtube_loader():
    url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
    save_dir="data/youtube/"
    loader = GenericLoader(
        YoutubeAudioLoader([url],save_dir),
        OpenAIWhisperParser()
    )
    docs = loader.load()
    print(docs[0].page_content[0:500])

# Document splitting

def document_splitting(document):
    chunk_size = 26
    chunk_overlap = 4
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    character_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return recursive_splitter, character_splitter


if __name__ == "main":
    recursive_splitter, character_splitter = document_splitting

