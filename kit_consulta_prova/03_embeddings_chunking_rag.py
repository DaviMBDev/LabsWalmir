"""
================================================================================
03 — EMBEDDINGS, CHUNKING (FIXO vs RECURSIVO) E RAG MANUAL COM GROQ (Labs 06-07)
================================================================================
Contém:
  1) Embeddings locais com SentenceTransformer ('paraphrase-multilingual-MiniLM-L12-v2' -> 384 dims)
  2) Similaridade de cosseno (cosine_similarity: MAIOR = MAIS SIMILAR, varia de -1 a 1)
  3) Comparação entre Chunking Fixo (CharacterTextSplitter) e Recursivo (RecursiveCharacterTextSplitter)
  4) Mini Busca Semântica em memória
  5) RAG completo manual com Groq + Grounding Estrito no prompt + Citação de fonte
"""

import os
import pdfplumber
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from groq import Groq

# ==============================================================================
# 1. CONFIGURAÇÃO GROQ + MODELO DE EMBEDDING
# ==============================================================================
load_dotenv(find_dotenv(), override=True)
GROQ_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_2")
# ⚠️ Atenção: 'llama-3.3-70b-versatile' deu 404 na conta atual. Use 'openai/gpt-oss-120b'.
MODELO_LLM = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MODELO_EMB_NOME = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 dimensões, ótimo em PT-BR


def listar_modelos_groq_ativos():
    """Comando salva-vidas se algum modelo der 404 na hora da prova!"""
    client = Groq(api_key=GROQ_KEY)
    return sorted(m.id for m in client.models.list().data)


# ==============================================================================
# 2. LEITURA DO PDF (TEXTO NATURAL — NUNCA LEMATIZAR PARA EMBEDDINGS/RAG!)
# ==============================================================================
def ler_pdf_natural(caminho: str) -> str:
    texto = ""
    with pdfplumber.open(caminho) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                texto += t + "\n"
    return texto.strip()


# ==============================================================================
# 3. CHUNKING: TAMANHO FIXO vs RECURSIVO (Lab 07)
# ==============================================================================
def comparar_chunking(texto: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """
    - Fixo (CharacterTextSplitter): corta rigidamente a cada N caracteres -> pode quebrar palavras no meio.
    - Recursivo (RecursiveCharacterTextSplitter): tenta separadores naturais (\n\n -> \n -> '. ' -> ' ')
      antes de cortar -> preserva parágrafos e frases (É O PADRÃO DE MERCADO).
    - chunk_overlap: sobreposição entre chunks vizinhos para não perder sentido nas bordas.
    """
    splitter_fixo = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splitter_recursivo = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks_fixo = splitter_fixo.split_text(texto)
    chunks_recursivo = splitter_recursivo.split_text(texto)
    return chunks_fixo, chunks_recursivo


# ==============================================================================
# 4. BUSCA SEMÂNTICA EM MEMÓRIA (Lab 06)
# ==============================================================================
def buscar_semantica_memoria(pergunta: str, chunks: list[str], emb_chunks, modelo_emb, top_k: int = 3):
    """
    1) Transforma a pergunta em vetor (encode)
    2) Calcula cosine_similarity contra todos os chunks
    3) Ordena do MAIOR score para o MENOR e retorna top_k pares (score, chunk)
    """
    emb_pergunta = modelo_emb.encode([pergunta])
    sims = cosine_similarity(emb_pergunta, emb_chunks)[0]
    melhores_idx = sims.argsort()[::-1][:top_k]
    return [(float(sims[i]), chunks[i]) for i in melhores_idx]


# ==============================================================================
# 5. RAG MANUAL COMPLETO COM GROUNDING ESTRITO (Lab 07)
# ==============================================================================
def responder_com_rag_manual(pergunta: str, chunks: list[str], emb_chunks, modelo_emb, top_k: int = 3) -> dict:
    # ETAPA 1 — RECUPERAR (Retrieval)
    recuperados = buscar_semantica_memoria(pergunta, chunks, emb_chunks, modelo_emb, top_k=top_k)
    contexto = "\n\n".join([f"[Trecho {idx+1} | sim={score:.2f}]: {txt}" for idx, (score, txt) in enumerate(recuperados)])

    # ETAPA 2 — AUMENTAR O PROMPT (Augmentation + Grounding Estrito)
    prompt = f"""Você é um assistente acadêmico confiável.
Responda à pergunta usando EXCLUSIVAMENTE as informações do CONTEXTO abaixo.
Cite de qual trecho veio a informação.
Se a resposta NÃO estiver no contexto, diga claramente: "Não encontrei essa informação no documento."

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}
"""

    # ETAPA 3 — GERAR (Generation via Groq)
    client = Groq(api_key=GROQ_KEY)
    resp = client.chat.completions.create(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # temperatura baixa = mais factual, menos criativo/alucinatório
    )
    return {
        "pergunta": pergunta,
        "resposta": resp.choices[0].message.content,
        "melhor_similaridade": round(recuperados[0][0], 3),
        "fontes": recuperados,
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    texto = ler_pdf_natural(os.path.join(base_dir, "introducaoml.pdf"))
    c_fixo, c_rec = comparar_chunking(texto, chunk_size=350, chunk_overlap=60)
    print(f"Chunks fixo: {len(c_fixo)} | Chunks recursivo: {len(c_rec)}")

    modelo_emb = SentenceTransformer(MODELO_EMB_NOME)
    emb_chunks = modelo_emb.encode(c_rec)
    print("Dimensões:", emb_chunks.shape)

    res = responder_com_rag_manual("O que é aprendizado supervisionado?", c_rec, emb_chunks, modelo_emb, top_k=2)
    print("Resposta:", res["resposta"])
