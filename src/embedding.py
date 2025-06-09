# Loading the PDF, chunking and embedding

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

def PDFLoadText(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return " ".join(page.extract_text() for page in reader.pages)

def ChunkText(text: str, size: int=500, overlap: int=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size-overlap):
        chunks.append(" ".join(words[i:i + size]))
    
    return chunks

def embed_chunks(chunks: list[str], model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(chunks, show_progress_bar=True)


# Testing using a sample PDF file
pdf_path = "data\\study_guide.pdf"
print(PDFLoadText(pdf_path=pdf_path))