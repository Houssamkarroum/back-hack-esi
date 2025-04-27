import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from deep_translator import GoogleTranslator
from dotenv import load_dotenv


load_dotenv()


# 1. Load and translate documents
def process_documents():
    loader = DirectoryLoader('data/', glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
   
    # Translate Arabic → English
    # translated_docs = []
    # for doc in docs:
    #     try:
    #         translated = GoogleTranslator(source='auto', target='en').translate(doc.page_content)
    #         doc.page_content = translated
    #         translated_docs.append(doc)
    #     except Exception as e:
    #         print(f"Error translating {doc.metadata.get('source', '<unknown>')}: {e}")
    return docs


# 2. Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# 3. Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# 4. Store in FAISS
def create_index():
    documents = process_documents()
    texts = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local("faiss_medical_index")


if __name__ == "__main__":


    create_index()



