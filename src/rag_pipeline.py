# LLM setup
import os
from typing import List
from vector_store import LoadVectorDB
from retriever import GetKChunks

from langchain_core.documents import Document
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def LoadLLM(model_path: str, temp: float, max_tokens: int, top_p: float, n_gpu_layers: int = 0):
    return LlamaCpp(model_path=model_path, temperature=temp, max_tokens=max_tokens, top_p=top_p, n_gpu_layers=n_gpu_layers, verbose=False)

def BuildPrompt():
    template = """
        You are a helpful assistant answering questions based on context.

        Context: {context}
        Question: {question}

        Answer:"""
    
    return PromptTemplate.from_template(template)

def RunRAG(machine, query:str, chunks: List[Document]):
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    prompt = BuildPrompt().format(context=context, question=query)
    return machine.invoke(prompt)

# Local testing

vec_db = LoadVectorDB(path="data\\vec_db")
query = "Define celerity"
top_suggestions = GetKChunks(query=query, vec_db=vec_db, k=3)

llm = LoadLLM(model_path="models\\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
              temp=0.65,
              max_tokens=512,
              top_p=0.90)

response = RunRAG(llm, query=query, chunks=top_suggestions)

print(f"\nQuery: {query}")
print(f"Answer: {response}")
