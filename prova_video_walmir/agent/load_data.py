import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def carregar_pdf(caminho_pdf: str) -> str:
    """Extrai o texto natural de um PDF usando PyMuPDF (fitz)."""
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {caminho_pdf}")
    doc = fitz.open(caminho_pdf)
    texto = " ".join(pagina.get_text() for pagina in doc)
    doc.close()
    return texto.strip()


def dividir_em_chunks(texto: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """Divide o texto em chunks usando estratégia recursiva."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(texto)
