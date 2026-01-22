import os
from langchain_community.document_loaders import PyPDFLoader

print("Current working directory:", os.getcwd())


path = r"C:\\RAG\\book.pdf"


def load_documents():
    docs = []
    if os.path.exists(path):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
        print('Document Loaded Successfully!')
    else:
        print(f"Error: File not found at {path}")
    return docs
# Call the function
documents = load_documents()
