import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from load_data import carregar_pdf, dividir_em_chunks
from vector_store import obter_colecao


def executar_ingestao():
    caminho_pdf = os.path.join(CURRENT_DIR, "data", "inteligencia artificial.pdf")
    texto = carregar_pdf(caminho_pdf)
    chunks = dividir_em_chunks(texto, chunk_size=400, chunk_overlap=80)

    col = obter_colecao()
    if col.count() == 0:
        col.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"origem": "inteligencia artificial.pdf", "indice": i} for i in range(len(chunks))],
        )
        print(f"Ingestão concluída: {len(chunks)} chunks indexados em data/chroma.")
    else:
        print(f"Coleção já possui {col.count()} chunks indexados. Reindexação ignorada.")


if __name__ == "__main__":
    executar_ingestao()
