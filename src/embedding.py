# Loading the PDF, chunking and embedding

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def PDFLoadText(pdf_path: str):
    reader = PyPDFLoader(pdf_path)
    return reader.load()

def ChunkText(docs, size: int=500, overlap: int=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def embed_chunks(chunks: list[str], model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vec_store = FAISS.from_documents(chunks, embedding=embeddings)
    return vec_store



# Testing using a sample PDF file
pdf_path = "data\\study_guide.pdf"

