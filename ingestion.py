import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore  

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("/Users/medivh/Library/Mobile Documents/com~apple~CloudDocs/Dev/langchain-course/mediumblog1.txt")
    documents = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

    print("ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    print("finish")
