"""
================================================================================
06 — AGENTE SUPERVISOR MULTI-AGENTE COM LANGGRAPH (MODELO DA PROVA PASSADA!)
================================================================================
Reproduz exatamente a arquitetura exibida no vídeo da prova anterior:
  - `ChatGroq` (langchain-groq)
  - `@tool` (langchain_core.tools)
  - `create_react_agent` (langgraph.prebuilt) -> cria subagentes especialistas
  - `create_supervisor` (langgraph_supervisor) -> orquestra delegação automática
  - `fitz` (PyMuPDF) para leitura de PDF

💡 COMO ADAPTAR EM 2 MINUTOS NA PROVA:
   1) Crie a nova função decorada com `@tool` e uma DOCSTRING super clara
      (o LangGraph gera o JSON Schema automaticamente a partir dos type hints + docstring!).
   2) Adicione a função na lista `tools=[...]` do subagente certo (ou crie um novo subagente).
   3) Atualize o `prompt` do `create_supervisor` explicando QUANDO delegar para esse subagente.
   4) Chame `supervisor.invoke({"messages": [{"role": "user", "content": "..."}]})`.
"""

import os
import fitz  # PyMuPDF
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

load_dotenv(find_dotenv(), override=True)
# ⚠️ No vídeo original era "llama-3.3-70b-versatile" (deu 404 na conta atual).
MODELO = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
llm = ChatGroq(model=MODELO, temperature=0)


# ==============================================================================
# 1. FERRAMENTAS (@tool) — DOCSTRING + TYPE HINTS SÃO OBRIGATÓRIOS!
# ==============================================================================
@tool
def buscar_na_web(query: str) -> str:
    """Busca informações textuais sobre conceitos, fatos e definições de um tema."""
    resultados = {
        "python": "Python é uma linguagem criada por Guido van Rossum em 1991, muito usada em IA e ciência de dados.",
        "langgraph": "LangGraph é um framework para construir sistemas multi-agente stateful como grafos.",
        "groq": "Groq oferece inferência ultrarrápida de LLMs via API.",
        "default": f"Resultado para '{query}': Machine Learning são algoritmos que aprendem padrões para prever e decidir.",
    }
    chave = next((k for k in resultados if k in query.lower()), "default")
    return resultados[chave]


@tool
def resumir_texto(texto: str) -> str:
    """Resume um texto longo em até 50 palavras. Útil após obter informações extensas."""
    palavras = texto.split()
    if len(palavras) <= 50:
        return texto
    return " ".join(palavras[:50]) + "..."


@tool
def extrair_e_contar_pdf(caminho_pdf: str) -> str:
    """Extrai o texto de um arquivo PDF e retorna total de páginas, palavras e caracteres."""
    try:
        if not os.path.exists(caminho_pdf):
            return f"Erro: Arquivo não encontrado: {caminho_pdf}"
        doc = fitz.open(caminho_pdf)
        paginas = len(doc)
        texto = "".join(pagina.get_text() for pagina in doc)
        doc.close()
        return (
            f"PDF lido com sucesso. Páginas: {paginas} | "
            f"Palavras: {len(texto.split())} | Total de caracteres: {len(texto)}"
        )
    except Exception as e:
        return f"Erro ao processar o PDF: {str(e)}"


@tool
def calcular(expressao: str) -> str:
    """Avalia uma expressão matemática segura (ex: '150 * 3 + 45'). Use para qualquer cálculo."""
    try:
        permitidos = set("0123456789+-*/(). ")
        if not set(expressao).issubset(permitidos):
            return "Erro: expressão contém caracteres não permitidos."
        resultado = eval(expressao, {"__builtins__": {}}, {})
        return f"Resultado de '{expressao}' = {resultado}"
    except Exception as e:
        return f"Erro ao calcular: {str(e)}"


# ==============================================================================
# 2. SUBAGENTES ESPECIALISTAS (create_react_agent)
# ==============================================================================
subagente_pesquisa = create_react_agent(
    model=llm,
    tools=[buscar_na_web, resumir_texto],
    name="subagente_pesquisa",
    prompt="Você é especialista em pesquisa e resumo. Use buscar_na_web e resumir_texto.",
)

subagente_pln = create_react_agent(
    model=llm,
    tools=[extrair_e_contar_pdf],
    name="subagente_pln",
    prompt="Você é especialista em PLN e PDFs. Use extrair_e_contar_pdf passando o caminho exato do PDF.",
)

subagente_calculo = create_react_agent(
    model=llm,
    tools=[calcular],
    name="subagente_calculo",
    prompt="Você é especialista em matemática. Sempre use a ferramenta calcular.",
)


# ==============================================================================
# 3. AGENTE SUPERVISOR (create_supervisor)
# ==============================================================================
supervisor = create_supervisor(
    agents=[subagente_pesquisa, subagente_pln, subagente_calculo],
    model=llm,
    prompt=(
        "Você é um Agente Supervisor que coordena três subagentes especialistas:\n"
        "- 'subagente_pesquisa': perguntas conceituais e resumos.\n"
        "- 'subagente_pln': leitura, extração de texto e contagem de caracteres/palavras de arquivos PDF.\n"
        "- 'subagente_calculo': operações e cálculos matemáticos.\n"
        "Delegue ao especialista correto e responda sempre em português de forma clara."
    ),
).compile()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, "introducaoml.pdf")

    print("--- Iniciando Agente Supervisor ---")
    res = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Analise o arquivo PDF em '{pdf_path}' e informe quantos caracteres e palavras ele possui.",
                }
            ]
        }
    )
    print("\nRESPOSTA FINAL:")
    print(res["messages"][-1].content)
