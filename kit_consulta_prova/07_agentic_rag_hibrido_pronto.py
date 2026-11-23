"""
================================================================================
07 — TEMPLATE CORINGA: AGENTIC RAG HÍBRIDO (PDF + ChromaDB + CSV + ReAct + Guardrails)
================================================================================
Este arquivo junta o MELHOR de todos os Labs (05, 06, 07, 08 e 09) em um único
template pronto para copiar e adaptar se a prova pedir integração completa:
  - Base Estruturada (CSV via Pandas) -> ferramentas de consulta/filtro/cálculo
  - Base Não Estruturada (PDF via ChromaDB) -> ferramenta de busca semântica com guardrail de distância
  - Agente ReAct (Groq) decide SOZINHO qual fonte consultar (Agentic RAG!)
  - Guardrails em 4 camadas:
      1) Entrada (pergunta curta/vazia)
      2) Grounding numérico na ferramenta RAG (`scores[0] < LIMIAR`)
      3) Regra de negócio nas ferramentas (`0 <= desconto <= 30`, arquivo/item inexistente)
      4) Loop ReAct (`max_iter`, whitelist `FUNCOES`, `try/except`, escopo no `SISTEMA`)
"""

import os
import json
import pandas as pd
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# ==============================================================================
# 1. CONFIGURAÇÃO GERAL
# ==============================================================================
load_dotenv(find_dotenv(), override=True)
GROQ_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_2")
MODELO = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

client_groq = Groq(api_key=GROQ_KEY)

# Carrega CSV (base estruturada)
df = pd.read_csv(os.path.join(BASE_DIR, "CarPrice_Assignment.csv"))
df["modelo"] = df["CarName"].str.lower().str.strip()

# Configura ChromaDB persistente (base não estruturada / PDF)
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_agentic_rag")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
colecao = chroma_client.get_or_create_collection(name="apostila_ml", embedding_function=emb_fn)


def garantir_pdf_indexado(caminho_pdf: str):
    if colecao.count() > 0:
        return
    texto = ""
    with pdfplumber.open(caminho_pdf) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                texto += t + "\n"
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=350,
        chunk_overlap=60,
    )
    chunks = splitter.split_text(texto.strip())
    colecao.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])


garantir_pdf_indexado(os.path.join(BASE_DIR, "introducaoml.pdf"))


# ==============================================================================
# 2. FERRAMENTAS HÍBRIDAS (CSV + RAG CHROMADB)
# ==============================================================================
def consultar_carro(modelo: str) -> dict:
    achados = df[df["modelo"].str.contains(modelo.lower().strip(), regex=False)]
    if achados.empty:
        return {"erro": f"modelo '{modelo}' não encontrado"}
    return {
        "modelos": sorted(achados["modelo"].unique())[:5],
        "preco_medio_usd": round(float(achados["price"].mean()), 2),
        "carroceria": achados["carbody"].mode()[0],
        "potencia_media_hp": int(achados["horsepower"].mean()),
    }


def aplicar_desconto(preco: float, percentual: float) -> dict:
    if not 0 <= float(percentual) <= 30:
        return {"erro": "desconto permitido apenas entre 0% e 30%"}
    return {"preco_final_usd": round(float(preco) * (1 - float(percentual) / 100), 2)}


def consultar_apostila_ml(pergunta: str, k: int = 2, limiar: float = 1.25) -> dict:
    """Ferramenta de Agentic RAG: consulta o ChromaDB com guardrail numérico de distância."""
    res = colecao.query(query_texts=[pergunta], n_results=k)
    docs = res["documents"][0]
    dists = res["distances"][0]
    if not dists or dists[0] >= limiar:
        return {
            "erro": "Informação não encontrada na apostila de Machine Learning.",
            "menor_distancia": round(float(dists[0]), 3) if dists else None,
        }
    return {
        "trechos_encontrados": docs,
        "menor_distancia": round(float(dists[0]), 3),
    }


ferramentas = [
    {
        "type": "function",
        "function": {
            "name": "consultar_carro",
            "description": "Consulta preço médio em USD, carroceria e potência de um modelo de carro no CSV.",
            "parameters": {
                "type": "object",
                "properties": {"modelo": {"type": "string", "description": "nome do modelo em minúsculas"}},
                "required": ["modelo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aplicar_desconto",
            "description": "Calcula preço final aplicando desconto de 0% a 30%.",
            "parameters": {
                "type": "object",
                "properties": {
                    "preco": {"type": "number"},
                    "percentual": {"type": "number"},
                },
                "required": ["preco", "percentual"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_apostila_ml",
            "description": (
                "Busca trechos na apostila PDF de Machine Learning (aprendizado supervisionado, "
                "não supervisionado, reforço, métricas, etc.). Use para qualquer dúvida teórica de ML."
            ),
            "parameters": {
                "type": "object",
                "properties": {"pergunta": {"type": "string", "description": "pergunta sobre Machine Learning"}},
                "required": ["pergunta"],
            },
        },
    },
]

FUNCOES = {
    "consultar_carro": consultar_carro,
    "aplicar_desconto": aplicar_desconto,
    "consultar_apostila_ml": consultar_apostila_ml,
}

SISTEMA = {
    "role": "system",
    "content": (
        "Você é um assistente híbrido especialista em carros (CSV) e na apostila de Machine Learning (PDF). "
        "Responda em português. Sempre use as ferramentas para preços, descontos e conceitos da apostila. "
        "Nunca invente dados. Recuse educadamente perguntas fora de carros e Machine Learning."
    ),
}


# ==============================================================================
# 3. AGENTE PROTEGIDO (ENTRADA + LOOP ReAct)
# ==============================================================================
def agente_hibrido(pergunta: str, max_iter: int = 5, verbose: bool = True) -> str:
    # Guardrail de entrada
    if not pergunta or len(pergunta.strip()) < 5:
        return "Pergunta muito curta ou vazia."

    mensagens = [SISTEMA, {"role": "user", "content": pergunta}]

    for i in range(max_iter):
        resp = client_groq.chat.completions.create(
            model=MODELO,
            messages=mensagens,
            tools=ferramentas,
            tool_choice="auto",
            temperature=0.1,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content

        mensagens.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            nome = tc.function.name
            func = FUNCOES.get(nome)
            args = {}
            try:
                args = json.loads(tc.function.arguments or "{}")
                obs = func(**args) if func else {"erro": f"ferramenta '{nome}' não autorizada"}
            except Exception as e:
                obs = {"erro": f"falha em {nome}: {e}"}

            if verbose:
                print(f"  [{i + 1}] {nome}({args}) -> {obs}")

            mensagens.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(obs, ensure_ascii=False, default=str),
                }
            )

    return "Não consegui concluir dentro do limite de passos."


if __name__ == "__main__":
    for q in [
        "Quanto custa o bmw 320i com 15% de desconto?",
        "Segundo a apostila, o que é aprendizado supervisionado?",
        "Qual a receita de bolo de chocolate?",
    ]:
        print(f"\n❓ {q}")
        print("🤖", agente_hibrido(q))
