# Loading the PDF, chunking and embedding

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def PDFLoadText(pdf_path: str):
    reader = PyPDFLoader(pdf_path)
    return reader.load()

def ChunkText(docs, size: int=300, overlap: int=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def EmbedChunks(chunks: list[str], model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vec_store = FAISS.from_documents(chunks, embedding=embeddings)
    return vec_store

# # Testing using a sample PDF file
# pdf_path = "data\\study_guide.pdf"

# raw_docs = PDFLoadText(pdf_path)
# chunks = ChunkText(raw_docs)
# vec_db = EmbedChunks(chunks)

# # print(len(chunks))

# # print(LoadVectorDB())