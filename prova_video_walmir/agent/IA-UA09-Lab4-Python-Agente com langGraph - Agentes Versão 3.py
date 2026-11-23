# ============================================================
# IA-UA09-Lab4 — Python Agente com LangGraph (Agentes Versão 3)
# Baseado fielmente nos trechos exibidos no vídeo da prova
# ============================================================

import os
from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# ------------------------------------------------------------
# 1. MODELO BASE
# ------------------------------------------------------------
# llama3-groq-70b-8192-tool-use-preview é otimizado para
# tool-calling no Groq - evita erros de JSON malformado.

# Nota: no vídeo original era "llama-3.3-70b-versatile" (atualizado para modelo ativo na conta Groq).
llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))


# ============================================================
# 2. FERRAMENTAS
# ============================================================

# ---------- Ferramentas do Subagente de Pesquisa ----------

@tool
def buscar_na_web(query: str) -> str:
    """
    Busca informações textuais sobre um tema.
    Use para perguntas sobre conceitos, fatos e definições.
    """
    resultados = {
        "python": "Python é uma linguagem criada por Guido van Rossum em 1991. Muito usada em IA, ciência de dados e web.",
        "langgraph": "LangGraph é um framework para construir sistemas multi-agente stateful, representando workflows como grafos.",
        "groq": "Groq é uma empresa de chips de IA que oferece inferência de LLMs de forma extremamente rápida e eficiente.",
        "default": f"Informação encontrada sobre '{query}': tópico com vasta literatura disponível.",
    }
    chave = next((k for k in resultados if k in query.lower()), "default")
    return resultados[chave]


# Ciclo da Função (tool) buscar_na_web:
# 1. Você pergunta: "O que é Python?"
#        ↓
# 2. LLM lê a docstring e decide:
#    "vou chamar buscar_na_web com query='Python'"
#        ↓
# 3. Python executa buscar_na_web("Python")
#    → retorna: "Python é uma linguagem criada por Guido van Rossum..."
#        ↓
# 4. Esse resultado é enviado de volta ao LLM
#        ↓
# 5. LLM usa o resultado para formular a resposta final ao usuário


@tool
def resumir_texto(texto: str) -> str:
    """
    Resume um texto longo em até 50 palavras.
    Útil após buscar informações extensas.
    """
    palavras = texto.split()
    if len(palavras) <= 50:
        return texto
    return " ".join(palavras[:50]) + "..."


# Ciclo da Função (tool) resumir_texto:
# 1. Você pergunta: "Me explique o que é Python resumidamente"
#        ↓
# 2. LLM lê a docstring de buscar_na_web e resumir_texto
#    Decide: "primeiro vou buscar, depois resumir"
#        ↓
# 3. Python executa buscar_na_web("Python")
#    → retorna: "Python é uma linguagem criada por Guido van Rossum
#      em 1991. Muito usada em IA, ciência de dados e web."
#        ↓
# 4. LLM recebe esse resultado
#    Decide: "o texto veio extenso, vou chamar resumir_texto"
#        ↓
# 5. Python executa resumir_texto("Python é uma linguagem...")
#    → conta as palavras: 18 palavras → menor que 50
#    → retorna o texto inteiro sem cortar
#        ↓
# 6. LLM recebe o texto resumido e formula a resposta final


# ---------- Ferramentas do Subagente de Cálculo ----------

@tool
def calcular(expressao: str) -> str:
    """
    Avalia uma expressão matemática simples (ex: '25 * 4 + 10').
    Use para qualquer cálculo numérico.
    """
    try:
        permitidos = set("0123456789+-*/(). ")
        if not set(expressao).issubset(permitidos):
            return "Erro: expressão contém caracteres não permitidos."
        resultado = eval(expressao, {"__builtins__": {}}, {})
        return f"Resultado de '{expressao}' = {resultado}"
    except Exception as e:
        return f"Erro ao calcular '{expressao}': {str(e)}"


# ============================================================
# 3. SUBAGENTES E SUPERVISOR
# ============================================================

subagente_pesquisa = create_react_agent(
    model=llm,
    tools=[buscar_na_web, resumir_texto],
    name="subagente_pesquisa",
    prompt="Você é um pesquisador. Use buscar_na_web e resumir_texto para responder dúvidas conceituais.",
)

subagente_calculo = create_react_agent(
    model=llm,
    tools=[calcular],
    name="subagente_calculo",
    prompt="Você é um especialista em cálculos matemáticos. Sempre use a ferramenta calcular.",
)

supervisor = create_supervisor(
    agents=[subagente_pesquisa, subagente_calculo],
    model=llm,
    prompt=(
        "Você é um supervisor que coordena dois subagentes: "
        "'subagente_pesquisa' (conceitos, definições e resumos) e "
        "'subagente_calculo' (operações matemáticas). "
        "Delegue corretamente e responda em português."
    ),
).compile()


if __name__ == "__main__":
    perguntas = [
        "Me explique o que é LangGraph resumidamente.",
        "Quanto é (150 * 3) + 45?",
    ]
    for p in perguntas:
        print(f"\n❓ {p}")
        res = supervisor.invoke({"messages": [{"role": "user", "content": p}]})
        print("🤖", res["messages"][-1].content)
