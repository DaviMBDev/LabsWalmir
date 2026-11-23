import os
import chromadb
from chromadb.utils import embedding_functions

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
CHROMA_PATH = os.path.join(PROJECT_DIR, "data", "chroma")


def obter_colecao(nome_colecao: str = "inteligencia_artificial"):
    """Retorna (ou cria) uma coleção persistente no ChromaDB em data/chroma."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    return client.get_or_create_collection(
        name=nome_colecao,
        embedding_function=emb_fn,
    )
