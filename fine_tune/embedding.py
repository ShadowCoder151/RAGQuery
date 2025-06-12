"""Optimization tasks"""

import logging
import os
import time

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def PDFLoadText(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        all_text = []

        for page in doc:
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))
            page_text = "\n".join(b[4].strip() for b in blocks if b[4].strip())
            all_text.append(page_text)

        print(page_text)
        doc.close()
        return "\n\n".join(all_text)

    except Exception as e:
        logging.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def ChunkText(docs, size: int=512, overlap: int=50):
    splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def EmbedChunks(chunks: list[str], model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vec_store = FAISS.from_documents(chunks, embedding=embeddings)
    return vec_store

pdf_path = "data\\study_guide.pdf"

st = time.time()
raw_docs = PDFLoadText(pdf_path)
print(raw_docs)

en = time.time()

print(f"Execution time is {en - st:.2f} ms")
# chunks = ChunkText(raw_docs)
# vec_db = EmbedChunks(chunks)

# # print(len(chunks))

# # print(LoadVectorDB())