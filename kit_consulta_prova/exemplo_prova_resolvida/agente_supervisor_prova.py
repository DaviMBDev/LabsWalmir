"""
REQUISITO 4 — Mesma Prova Resolvida em Arquitetura Multi-Agente Supervisor (LangGraph)
Exatamente no estilo do vídeo da prova do semestre passado!
"""

import os
import sys
import json
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tools_carros import (
    consultar_carro as _consultar_carro,
    buscar_por_orcamento as _buscar_por_orcamento,
    aplicar_desconto as _aplicar_desconto,
    comparar_carros as _comparar_carros,
    analisar_pdf as _analisar_pdf,
    consultar_apostila_ml as _consultar_apostila_ml,
    PDF_PADRAO,
)

load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
if not os.getenv("GROQ_API_KEY"):
    load_dotenv(find_dotenv(), override=True)

MODELO = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
llm = ChatGroq(model=MODELO, temperature=0)


# ==============================================================================
# 1. WRAPPERS @tool (LangChain/LangGraph)
# ==============================================================================
@tool
def consultar_carro(modelo: str) -> str:
    """Consulta preço médio em USD, carroceria, combustível e potência de um modelo de carro."""
    return json.dumps(_consultar_carro(modelo), ensure_ascii=False)


@tool
def buscar_por_orcamento(preco_max: float, carroceria: str = "", limite: int = 3) -> str:
    """Lista os carros mais baratos até preco_max em USD, filtrando opcionalmente por carroceria (sedan, hatchback, wagon, hardtop, convertible)."""
    return json.dumps(_buscar_por_orcamento(preco_max, carroceria or None, limite), ensure_ascii=False)


@tool
def aplicar_desconto(preco: float, percentual: float) -> str:
    """Calcula desconto entre 0% e 30% sobre um preço em USD."""
    return json.dumps(_aplicar_desconto(preco, percentual), ensure_ascii=False)


@tool
def comparar_carros(modelo_a: str, modelo_b: str) -> str:
    """Compara dois modelos de carros (preço médio, diferença, mais barato e mais potente)."""
    return json.dumps(_comparar_carros(modelo_a, modelo_b), ensure_ascii=False)


@tool
def extrair_e_contar_pdf(caminho_pdf: str = "") -> str:
    """Extrai texto de um PDF com PyMuPDF e retorna total de páginas, palavras e caracteres."""
    return json.dumps(_analisar_pdf(caminho_pdf or PDF_PADRAO), ensure_ascii=False)


@tool
def consultar_apostila_ml(pergunta: str) -> str:
    """Busca trechos relevantes na apostila de Machine Learning indexada no ChromaDB."""
    return json.dumps(_consultar_apostila_ml(pergunta), ensure_ascii=False)


# ==============================================================================
# 2. SUBAGENTES ESPECIALISTAS + SUPERVISOR
# ==============================================================================
subagente_carros = create_react_agent(
    model=llm,
    tools=[consultar_carro, buscar_por_orcamento, aplicar_desconto, comparar_carros],
    name="subagente_carros",
    prompt=(
        "Você é especialista na base de carros CarPrice (valores em USD). "
        "Use suas ferramentas para consultar, comparar, filtrar por orçamento e aplicar descontos (0% a 30%)."
    ),
)

subagente_pln_rag = create_react_agent(
    model=llm,
    tools=[extrair_e_contar_pdf, consultar_apostila_ml],
    name="subagente_pln_rag",
    prompt=(
        "Você é especialista em PLN e RAG sobre a apostila PDF de Machine Learning. "
        "Use extrair_e_contar_pdf para contar páginas/palavras/caracteres e consultar_apostila_ml para responder perguntas sobre o conteúdo."
    ),
)

supervisor = create_supervisor(
    agents=[subagente_carros, subagente_pln_rag],
    model=llm,
    prompt=(
        "Você é um Agente Supervisor que coordena dois subagentes:\n"
        "- 'subagente_carros': tudo sobre carros, preços, orçamento, comparações e descontos.\n"
        "- 'subagente_pln_rag': leitura/contagem do PDF e dúvidas conceituais da apostila de Machine Learning.\n"
        "Se a pergunta estiver fora desses dois domínios, recuse educadamente sem chamar subagentes. "
        "Responda sempre em português."
    ),
).compile()


if __name__ == "__main__":
    print("--- Iniciando Agente Supervisor (Prova Resolvida) ---")
    testes = [
        f"Leia o arquivo PDF '{PDF_PADRAO}' e informe quantas páginas, palavras e caracteres ele possui.",
        "Segundo a apostila, qual a diferença entre aprendizado supervisionado e não supervisionado?",
        "Qual o preço médio do bmw 320i e quanto fica com 10% de desconto?",
    ]
    for t in testes:
        print(f"\n❓ {t}")
        res = supervisor.invoke({"messages": [{"role": "user", "content": t}]})
        print("RESPOSTA FINAL:")
        print(res["messages"][-1].content)
