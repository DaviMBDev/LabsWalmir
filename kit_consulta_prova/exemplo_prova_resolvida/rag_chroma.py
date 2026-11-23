"""
REQUISITO 1 — Pipeline RAG com Chunking Recursivo, ChromaDB Persistente e Guardrails
Baseado nos Labs 05, 06, 07 e 08.
"""

import os
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PADRAO = os.path.join(BASE_DIR, "data", "introducaoml.pdf")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_prova")
MODELO_EMB = "paraphrase-multilingual-MiniLM-L12-v2"
LIMIAR_DISTANCIA = 1.25  # Menor distância = mais similar no ChromaDB


def ler_pdf_natural(caminho_pdf: str) -> str:
    """Extrai texto NATURAL do PDF (sem lematizar/remover stopwords, ideal para RAG)."""
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"PDF não encontrado: {caminho_pdf}")
    texto = ""
    with pdfplumber.open(caminho_pdf) as pdf:
        for pagina in pdf.pages:
            t = pagina.extract_text()
            if t:
                texto += t + "\n"
    return texto.strip()


def obter_colecao(nome_colecao: str = "apostila_ml_prova"):
    """Cria/carrega a coleção persistente no ChromaDB e garante a indexação 1 única vez."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODELO_EMB)
    col = client.get_or_create_collection(name=nome_colecao, embedding_function=emb_fn)

    # Proteção contra duplicação (Lab 08)
    if col.count() == 0:
        texto = ler_pdf_natural(PDF_PADRAO)
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=350,
            chunk_overlap=60,
        )
        chunks = splitter.split_text(texto)
        col.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"origem": os.path.basename(PDF_PADRAO), "indice": i} for i in range(len(chunks))],
        )
    return col


def guardrail_entrada(pergunta: str) -> tuple[bool, str | None]:
    """Guardrail 1: bloqueia entradas vazias ou curtas demais antes de gastar chamadas."""
    if not pergunta or len(pergunta.strip()) < 5:
        return False, "Pergunta muito curta ou vazia."
    return True, None


def contexto_relevante(scores: list[float], limiar: float = LIMIAR_DISTANCIA) -> bool:
    """Guardrail 2: grounding verificável por distância vetorial (menor = mais próximo)."""
    return len(scores) > 0 and scores[0] < limiar


def buscar_na_apostila(pergunta: str, k: int = 3, limiar: float = LIMIAR_DISTANCIA) -> dict:
    """Executa busca vetorial protegida no ChromaDB (usada tanto standalone quanto como tool)."""
    ok, msg = guardrail_entrada(pergunta)
    if not ok:
        return {"erro": msg, "guardrail": "entrada"}

    col = obter_colecao()
    res = col.query(query_texts=[pergunta], n_results=k)
    docs = res["documents"][0]
    scores = res["distances"][0]

    if not contexto_relevante(scores, limiar=limiar):
        return {
            "erro": "Não encontrei essa informação na apostila de Machine Learning.",
            "guardrail": "grounding",
            "menor_distancia": round(float(scores[0]), 3) if scores else None,
        }

    return {
        "trechos": docs,
        "distancias": [round(float(s), 3) for s in scores],
        "menor_distancia": round(float(scores[0]), 3),
    }


if __name__ == "__main__":
    col = obter_colecao()
    print(f"ChromaDB pronto! Chunks na coleção: {col.count()}")
    for q in ["O que é aprendizado supervisionado?", "oi", "Qual a receita de lasanha?"]:
        print(f"\nConsulta: {q}")
        print("Resultado:", buscar_na_apostila(q, k=2))
