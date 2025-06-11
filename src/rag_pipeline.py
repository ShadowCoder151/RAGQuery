# LLM setup
import os
from typing import List

from langchain_core.document_loaders import Document
from langchain_community.llms import llamacpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def LoadLLM(model_path: str, temp: float, max_tokens: int, top_p: float, n_gpu_layers: int = 0):
    pass

def BuildPrompt():
    pass

def RunRAG(machine, query:str, chunks: List[Document]):
    pass