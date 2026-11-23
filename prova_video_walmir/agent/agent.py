import os
import sys
from dotenv import load_dotenv

# Permite importar tools.py tanto rodando da raiz quanto de dentro de agent/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

load_dotenv(os.path.join(PROJECT_DIR, ".env"))

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from tools import buscar_na_web, resumir_texto, extrair_e_contar_pdf

# 1. Modelo base (Groq)
# Nota: no vídeo original era "llama-3.3-70b-versatile" (indisponível/404 na conta atual).
MODELO = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
llm = ChatGroq(model=MODELO, temperature=0)

# 2. Subagente de Pesquisa (busca e resumo)
subagente_pesquisa = create_react_agent(
    model=llm,
    tools=[buscar_na_web, resumir_texto],
    name="subagente_pesquisa",
    prompt=(
        "Você é um subagente especialista em pesquisa e resumo de textos. "
        "Use suas ferramentas para buscar informações conceituais e resumir textos longos."
    ),
)

# 3. Subagente de PLN / Leitura de PDF
subagente_pln = create_react_agent(
    model=llm,
    tools=[extrair_e_contar_pdf],
    name="subagente_pln",
    prompt=(
        "Você é um subagente especialista em PLN e análise de documentos PDF. "
        "Use a ferramenta extrair_e_contar_pdf passando o caminho exato do arquivo PDF."
    ),
)

# 4. Agente Supervisor orquestrando os subagentes
supervisor = create_supervisor(
    agents=[subagente_pesquisa, subagente_pln],
    model=llm,
    prompt=(
        "Você é um Agente Supervisor responsável por delegar tarefas aos subagentes:\n"
        "- Delegue para 'subagente_pesquisa' perguntas conceituais ou pedidos de resumo.\n"
        "- Delegue para 'subagente_pln' tarefas que envolvam ler, extrair texto ou contar caracteres de um arquivo PDF.\n"
        "Sempre responda em português de forma clara e objetiva com o resultado obtido."
    ),
).compile()


if __name__ == "__main__":
    arquivo_nome = "inteligencia artificial.pdf"
    caminho_pdf = os.path.join(CURRENT_DIR, "data", arquivo_nome)

    print("--- Iniciando Agente Supervisor ---")
    print(f"Analisando arquivo: {arquivo_nome}\n")

    resultado = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Leia o arquivo PDF localizado em '{caminho_pdf}' "
                        "e informe se foi lido com sucesso e qual o total de caracteres."
                    ),
                }
            ]
        }
    )

    print("RESPOSTA FINAL:")
    print(resultado["messages"][-1].content)
