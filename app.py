from fine_tune.vector_store import LoadVectorDB, SaveVectorDB
from fine_tune.retriever import GetKChunks
from fine_tune.embedding import PDFLoadText, ChunkText, EmbedChunks

# Testing using a sample PDF file
pdf_path = "data\\study_guide.pdf"

raw_docs = PDFLoadText(pdf_path)
chunks = ChunkText(raw_docs)
vec_db = EmbedChunks(chunks)

SaveVectorDB(vec_db=vec_db)

vec_db = LoadVectorDB(path="data\\vec_db")
query = "Define celerity"

top_chunks = GetKChunks(query=query, vec_db=vec_db, k=3)

for i, chunk in enumerate(top_chunks):
    print(f"[{i + 1}] {chunk.page_content[:1000]}....")
    print()