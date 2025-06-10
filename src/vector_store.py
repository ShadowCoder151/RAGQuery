# Vector Store Operations

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def SaveVectorDB(vec_db, path="data\\vec_db"):
    vec_db.save_local(path)

def LoadVectorDB(path="data\\vec_db", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(folder_path=path, embeddings=embeddings, allow_dangerous_deserialization=True)