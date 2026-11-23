"""
REQUISITO 3 — Agente ReAct Completo em Groq Puro (Sem Frameworks · Lab 09 + Agentic RAG)
Contém:
  - Catálogo JSON das 6 ferramentas (com `enum` em carroceria)
  - Whitelist `FUNCOES`
  - Prompt `SISTEMA` com anti-alucinação e guardrail de escopo
  - Loop ReAct `agente()` com 3 guardrails (max_iter, whitelist, try/except)
  - Bateria completa de 8 testes cobrindo todos os requisitos da prova
"""

import os
import sys
import json
from dotenv import load_dotenv, find_dotenv
from groq import Groq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tools_carros import (
    consultar_carro,
    buscar_por_orcamento,
    aplicar_desconto,
    comparar_carros,
    analisar_pdf,
    consultar_apostila_ml,
    PDF_PADRAO,
)

load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
if not os.getenv("GROQ_API_KEY"):
    load_dotenv(find_dotenv(), override=True)

GROQ_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_2")
MODELO = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
client = Groq(api_key=GROQ_KEY)

# ==============================================================================
# CATÁLOGO JSON DAS 6 FERRAMENTAS
# ==============================================================================
ferramentas = [
    {
        "type": "function",
        "function": {
            "name": "consultar_carro",
            "description": "Consulta um carro pelo modelo (ex: 'bmw 320i', 'toyota corolla') e retorna preço médio em USD, carroceria, combustível e potência.",
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
            "name": "buscar_por_orcamento",
            "description": "Lista os carros mais baratos até um preço máximo em dólares, com filtro opcional de carroceria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "preco_max": {"type": "number", "description": "orçamento máximo em USD"},
                    "carroceria": {
                        "type": "string",
                        "enum": ["sedan", "hatchback", "wagon", "hardtop", "convertible"],
                    },
                    "limite": {"type": "integer", "description": "máximo de opções (padrão 3)"},
                },
                "required": ["preco_max"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aplicar_desconto",
            "description": "Calcula o preço final aplicando desconto percentual (regra: 0% a 30%). Use SEMPRE para descontos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "preco": {"type": "number"},
                    "percentual": {"type": "number", "description": "ex.: 15 para 15%"},
                },
                "required": ["preco", "percentual"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "comparar_carros",
            "description": "Compara dois modelos de carros (preço médio, diferença de preço, qual é mais barato e qual é mais potente).",
            "parameters": {
                "type": "object",
                "properties": {
                    "modelo_a": {"type": "string"},
                    "modelo_b": {"type": "string"},
                },
                "required": ["modelo_a", "modelo_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analisar_pdf",
            "description": "Lê o arquivo PDF da apostila e informa quantas páginas, palavras e caracteres ele possui.",
            "parameters": {
                "type": "object",
                "properties": {
                    "caminho_pdf": {"type": "string", "description": "caminho opcional do arquivo PDF"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_apostila_ml",
            "description": "Busca trechos na apostila de Machine Learning (ChromaDB) para responder dúvidas conceituais sobre ML.",
            "parameters": {
                "type": "object",
                "properties": {"pergunta": {"type": "string"}},
                "required": ["pergunta"],
            },
        },
    },
]

# WHITELIST
FUNCOES = {
    "consultar_carro": consultar_carro,
    "buscar_por_orcamento": buscar_por_orcamento,
    "aplicar_desconto": aplicar_desconto,
    "comparar_carros": comparar_carros,
    "analisar_pdf": analisar_pdf,
    "consultar_apostila_ml": consultar_apostila_ml,
}

# PROMPT DE SISTEMA COM GUARDRAIL DE ESCOPO
SISTEMA = {
    "role": "system",
    "content": (
        "Você é um assistente oficial da FATEC especialista em dois domínios: "
        "(1) consultoria de veículos da base CarPrice (preços em USD) e "
        "(2) dúvidas e análise da apostila de Machine Learning (PDF). "
        "Responda sempre em português de forma clara e objetiva. "
        "Use SEMPRE as ferramentas para preços, buscas, descontos, comparações, estatísticas de PDF e conteúdo da apostila — NUNCA invente valores. "
        "Se uma ferramenta retornar erro, explique o motivo ao usuário. "
        "GUARDRAIL DE ESCOPO: se o usuário perguntar sobre qualquer assunto fora de carros ou Machine Learning "
        "(ex.: capitais de países, receitas, esportes), recuse educadamente sem responder o tema fora do escopo."
    ),
}


# ==============================================================================
# LOOP ReAct COM OS 3 GUARDRAILS
# ==============================================================================
def agente(pergunta: str, max_iter: int = 5, verbose: bool = True) -> str:
    # Guardrail de entrada
    if not pergunta or len(pergunta.strip()) < 5:
        return "Pergunta muito curta ou vazia."

    mensagens = [SISTEMA, {"role": "user", "content": pergunta}]

    for i in range(max_iter):  # Guardrail 1: limite de passos
        resp = client.chat.completions.create(
            model=MODELO,
            messages=mensagens,
            tools=ferramentas,
            tool_choice="auto",
            temperature=0.1,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:  # Resposta final!
            return msg.content

        mensagens.append(msg.model_dump(exclude_none=True))

        for tc in msg.tool_calls:
            nome = tc.function.name
            func = FUNCOES.get(nome)  # Guardrail 2: whitelist
            args = {}
            try:
                args = json.loads(tc.function.arguments or "{}")
                obs = func(**args) if func else {"erro": f"ferramenta '{nome}' não autorizada"}
            except Exception as e:  # Guardrail 3: erro de argumentos/execução
                obs = {"erro": f"falha ao executar {nome}: {e}"}

            if verbose:
                print(f"  [passo {i + 1}] {nome}({args}) -> {obs}")

            mensagens.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(obs, ensure_ascii=False, default=str),
                }
            )

    return "Não consegui concluir dentro do limite de passos."


if __name__ == "__main__":
    bateria = [
        ("1. Multi-step ReAct (2 ferramentas)", "Tenho até 9 mil dólares e quero um sedan. Quais as opções e quanto fica a mais barata com 15% de desconto?"),
        ("2. Comparação de carros", "Compare o toyota corolla com o honda civic."),
        ("3. Guardrail de negócio (desconto 80%)", "Me dá 80% de desconto no bmw 320i."),
        ("4. Carro inexistente", "Quanto custa um fusca?"),
        ("5. Guardrail de escopo", "Qual a capital da França?"),
        ("6. Análise de PDF (estilo prova passada)", f"Analise o PDF '{PDF_PADRAO}' e informe páginas, palavras e caracteres."),
        ("7. Agentic RAG (ChromaDB)", "Segundo a apostila, quais são os principais tipos de aprendizado de máquina?"),
    ]

    print("=== EXECUÇÃO DA PROVA RESOLVIDA (AGENTE ReAct GROQ PURO) ===")
    for titulo, p in bateria:
        print(f"\n--- {titulo} ---")
        print("❓", p)
        print("🤖", agente(p))

    print("\n--- 8. Teste de Guardrail de Passos (max_iter=1) ---")
    print("🤖", agente(bateria[0][1], max_iter=1))
