# Query embedding and top -k retrieval

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def QueryEmbed(query: str, model_name= "sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings.embed_query(query)

def GetKChunks(query:str, vec_db: FAISS, k=10):
    return vec_db.similarity_search(query, k=k)

