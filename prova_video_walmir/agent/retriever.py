import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from vector_store import obter_colecao


def buscar_contexto(pergunta: str, k: int = 3):
    """Recupera os k chunks mais próximos da pergunta no ChromaDB."""
    col = obter_colecao()
    res = col.query(query_texts=[pergunta], n_results=k)
    docs = res["documents"][0]
    dists = res["distances"][0]
    return docs, dists
