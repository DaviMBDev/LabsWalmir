"""
================================================================================
04 — RAG COM CHROMADB PERSISTENTE E GUARDRAILS (Lab 08)
================================================================================
Contém:
  1) ChromaDB PersistentClient + SentenceTransformerEmbeddingFunction
  2) Indexação idempotente (`if col.count() == 0:`) para NÃO duplicar documentos
  3) Recuperação com `col.query()` (⚠️ MENOR distância = MAIS similar!)
  4) Guardrail 1 — Entrada (barra pergunta vazia/curta antes de gastar chamada ao LLM)
  5) Guardrail 2 — Grounding numérico por limiar (`scores[0] < LIMIAR`)
  6) Pipeline completo `responder_protegido()` pronto para copiar/adaptar
"""

import os
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
from groq import Groq

load_dotenv(find_dotenv(), override=True)
GROQ_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_2")
MODELO_LLM = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MODELO_EMB_NOME = "paraphrase-multilingual-MiniLM-L12-v2"

# Limiar máximo de distância vetorial aceitável (ajuste olhando as distâncias reais nos testes!)
# No ChromaDB (distância L2 padrão):
#   - Pergunta DENTRO do PDF costuma dar distância ~0.4 a ~1.1
#   - Pergunta FORA do PDF costuma dar distância > 1.3
LIMIAR_DISTANCIA = 1.25


# ==============================================================================
# 1. CRIAR / CARREGAR BANCO VETORIAL PERSISTENTE (ChromaDB)
# ==============================================================================
def criar_colecao_chroma(pasta_chroma: str = "./chroma_consulta", nome_colecao: str = "base_ml"):
    client = chromadb.PersistentClient(path=pasta_chroma)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODELO_EMB_NOME
    )
    col = client.get_or_create_collection(
        name=nome_colecao,
        embedding_function=emb_fn,
    )
    return col


# ==============================================================================
# 2. INDEXAR PDF NO CHROMA (COM PROTEÇÃO CONTRA DUPLICAÇÃO!)
# ==============================================================================
def indexar_pdf_no_chroma(col, caminho_pdf: str, chunk_size: int = 400, chunk_overlap: int = 80):
    if col.count() > 0:
        print(f"Coleção já indexada ({col.count()} chunks). Pulando reindexação.")
        return

    texto = ""
    with pdfplumber.open(caminho_pdf) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                texto += t + "\n"

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(texto.strip())

    col.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"origem": os.path.basename(caminho_pdf), "chunk_id": i} for i in range(len(chunks))],
    )
    print(f"Indexados {len(chunks)} chunks de '{caminho_pdf}' no ChromaDB.")


# ==============================================================================
# 3. RECUPERAR DO CHROMA (col.query)
# ==============================================================================
def recuperar(col, pergunta: str, k: int = 3):
    res = col.query(query_texts=[pergunta], n_results=k)
    docs = res["documents"][0]
    scores = res["distances"][0]  # ⚠️ ATENÇÃO: MENOR distância = MAIS similar!
    return docs, scores


# ==============================================================================
# 4. GUARDRAILS (ENTRADA + GROUNDING NUMÉRICO)
# ==============================================================================
def guardrail_entrada(pergunta: str) -> tuple[bool, str | None]:
    """GUARDRAIL 1: filtra entradas vazias ou curtas demais ANTES de chamar o LLM."""
    if not pergunta or len(pergunta.strip()) < 5:
        return False, "Pergunta muito curta ou vazia."
    return True, None


def contexto_relevante(scores: list[float], limiar: float = LIMIAR_DISTANCIA) -> bool:
    """GUARDRAIL 2: só libera a resposta se o melhor chunk (scores[0]) for próximo o bastante."""
    return len(scores) > 0 and scores[0] < limiar


# ==============================================================================
# 5. PIPELINE COMPLETO: ENTRADA -> CHROMA -> GROUNDING -> GROQ
# ==============================================================================
def responder_protegido(col, pergunta: str, limiar: float = LIMIAR_DISTANCIA) -> str:
    # 1) Guardrail de Entrada
    ok, msg = guardrail_entrada(pergunta)
    if not ok:
        return f"[GUARDRAIL ENTRADA] {msg}"

    # 2) Recuperar no ChromaDB
    docs, scores = recuperar(col, pergunta, k=3)

    # 3) Guardrail de Grounding Numérico
    if not contexto_relevante(scores, limiar=limiar):
        return (
            f"[GUARDRAIL GROUNDING] Não encontrei essa informação na base de conhecimento "
            f"(menor distância = {scores[0]:.3f} >= limiar {limiar})."
        )

    # 4) Gerar resposta fundamentada no Groq
    client_groq = Groq(api_key=GROQ_KEY)
    contexto = "\n\n".join(docs)
    prompt = f"""Responda em português usando APENAS o contexto abaixo.
Cite a ideia principal encontrada no texto.
Se não estiver no contexto, diga que não sabe.

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}
"""
    resp = client_groq.chat.completions.create(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    col = criar_colecao_chroma(os.path.join(base_dir, "chroma_kit_teste"))
    indexar_pdf_no_chroma(col, os.path.join(base_dir, "introducaoml.pdf"))

    for q in [
        "O que é aprendizado de máquina?",
        "oi",
        "Qual a receita de lasanha à bolonhesa?",
    ]:
        print(f"\n❓ {q}")
        print("🤖", responder_protegido(col, q))
