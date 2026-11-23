"""
================================================================================
05 — AGENTE COM FUNCTION CALLING E LOOP ReAct EM GROQ PURO (Lab 09 — FOCO!)
================================================================================
Constrói do zero e SEM frameworks um agente completo com:
  1) Ferramentas Python reais (devolvem SEMPRE um dict, inclusive em erro!)
     - consultar_carro(modelo)
     - buscar_por_orcamento(preco_max, carroceria, limite)
     - aplicar_desconto(preco, percentual)   <- GUARDRAIL DE NEGÓCIO (0% a 30%)
     - comparar_carros(modelo_a, modelo_b)   <- DESAFIO EXTRA DO LAB 09
  2) Catálogo JSON (`ferramentas`) com `description` clara e `enum` (GUARDRAIL DE SCHEMA)
  3) Whitelist `FUNCOES` (nome string -> função Python real)
  4) Prompt `SISTEMA` com anti-alucinação e GUARDRAIL DE ESCOPO
  5) Loop ReAct `agente()` com 3 GUARDRAILS DO AGENTE:
     - Guardrail 1: limite de passos (`max_iter`)
     - Guardrail 2: lista branca (`FUNCOES.get(nome)`)
     - Guardrail 3: tratamento de exceções (`try/except`)

💡 COMO ADAPTAR NA PROVA PARA QUALQUER OUTRO CSV/PROBLEMA:
   1. Troque o CSV e os nomes de colunas nas funções da Parte B.
   2. Atualize o catálogo JSON `ferramentas` e o dicionário `FUNCOES`.
   3. Atualize o papel e escopo no `SISTEMA`.
   4. O loop `agente()` NÃO MUDA NADA! É 100% reutilizável!
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# ==============================================================================
# PARTE A — AMBIENTE, CLIENTE GROQ E DADOS
# ==============================================================================
load_dotenv(find_dotenv(), override=True)
GROQ_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_2")
# ⚠️ 'llama-3.3-70b-versatile' deu 404 na conta atual. Use 'openai/gpt-oss-120b'.
MODELO = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

client = Groq(api_key=GROQ_KEY)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "CarPrice_Assignment.csv"))
df["modelo"] = df["CarName"].str.lower().str.strip()


# ==============================================================================
# PARTE B — FERRAMENTAS (FUNÇÕES PYTHON REAIS)
# Regra de ouro:
#   - O LLM NUNCA toca no DataFrame; ele só pede a função via JSON.
#   - Toda função devolve um dict simples (inclusive quando dá erro!).
#   - SEMPRE teste cada função isoladamente SEM o LLM antes de rodar o agente!
# ==============================================================================
def consultar_carro(modelo: str) -> dict:
    achados = df[df["modelo"].str.contains(modelo.lower().strip(), regex=False)]
    if achados.empty:
        return {"erro": f"modelo '{modelo}' não encontrado na base"}
    return {
        "modelos_encontrados": sorted(achados["modelo"].unique())[:5],
        "registros": int(len(achados)),
        "preco_medio_usd": round(float(achados["price"].mean()), 2),
        "carroceria": achados["carbody"].mode()[0],
        "combustivel": achados["fueltype"].mode()[0],
        "potencia_media_hp": int(achados["horsepower"].mean()),
    }


def buscar_por_orcamento(preco_max: float, carroceria: str = None, limite: int = 3) -> dict:
    base = df[df["price"] <= float(preco_max)]
    if carroceria:
        base = base[base["carbody"] == carroceria.lower().strip()]
    base = base.sort_values("price").drop_duplicates("modelo").head(int(limite))
    if base.empty:
        return {"erro": "nenhum carro encontrado com esses critérios"}
    return {"opcoes": base[["modelo", "carbody", "horsepower", "price"]].to_dict("records")}


def aplicar_desconto(preco: float, percentual: float) -> dict:
    # GUARDRAIL DE NEGÓCIO: impede descontos fora da faixa permitida
    if not 0 <= float(percentual) <= 30:
        return {"erro": "desconto permitido: de 0% a 30%"}
    final = float(preco) * (1 - float(percentual) / 100)
    return {
        "preco_original": float(preco),
        "percentual": float(percentual),
        "preco_final": round(final, 2),
    }


def comparar_carros(modelo_a: str, modelo_b: str) -> dict:
    """Desafio extra do Lab 09: compara dois modelos reutilizando consultar_carro()."""
    info_a = consultar_carro(modelo_a)
    if "erro" in info_a:
        return {"erro": f"Modelo A inválido: {info_a['erro']}"}
    info_b = consultar_carro(modelo_b)
    if "erro" in info_b:
        return {"erro": f"Modelo B inválido: {info_b['erro']}"}

    dif_preco = round(abs(info_a["preco_medio_usd"] - info_b["preco_medio_usd"]), 2)
    mais_barato = modelo_a if info_a["preco_medio_usd"] <= info_b["preco_medio_usd"] else modelo_b
    mais_potente = modelo_a if info_a["potencia_media_hp"] >= info_b["potencia_media_hp"] else modelo_b

    return {
        "modelo_a": {"consulta": modelo_a, **info_a},
        "modelo_b": {"consulta": modelo_b, **info_b},
        "diferenca_preco_usd": dif_preco,
        "mais_barato": mais_barato,
        "mais_potente": mais_potente,
    }


# ==============================================================================
# PARTE C — CATÁLOGO JSON (`ferramentas`), WHITELIST (`FUNCOES`) E `SISTEMA`
# Lembrete: a `description` É PROMPT! Quanto mais clara, melhor o LLM escolhe.
# ==============================================================================
ferramentas = [
    {
        "type": "function",
        "function": {
            "name": "consultar_carro",
            "description": (
                "Consulta um carro pelo nome do modelo (ex.: 'bmw 320i', 'toyota corolla') "
                "e retorna preço médio em dólares, carroceria, combustível e potência."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "modelo": {
                        "type": "string",
                        "description": "nome do modelo em minúsculas",
                    }
                },
                "required": ["modelo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_por_orcamento",
            "description": (
                "Lista os carros mais baratos até um preço máximo em dólares, "
                "opcionalmente filtrando pelo tipo de carroceria."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preco_max": {"type": "number", "description": "preço máximo em dólares"},
                    "carroceria": {
                        "type": "string",
                        # GUARDRAIL DE SCHEMA: restringe os valores aceitos!
                        "enum": ["sedan", "hatchback", "wagon", "hardtop", "convertible"],
                    },
                    "limite": {"type": "integer", "description": "quantidade de opções (padrão 3)"},
                },
                "required": ["preco_max"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aplicar_desconto",
            "description": (
                "Aplica um desconto percentual (0% a 30%) sobre um preço. "
                "Use SEMPRE para calcular descontos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preco": {"type": "number", "description": "preço original em dólares"},
                    "percentual": {"type": "number", "description": "ex.: 10 para 10%"},
                },
                "required": ["preco", "percentual"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "comparar_carros",
            "description": (
                "Compara dois modelos de carro lado a lado (preço médio, diferença de preço, "
                "mais barato e mais potente)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "modelo_a": {"type": "string", "description": "primeiro modelo em minúsculas"},
                    "modelo_b": {"type": "string", "description": "segundo modelo em minúsculas"},
                },
                "required": ["modelo_a", "modelo_b"],
            },
        },
    },
]

# WHITELIST: o agente só pode executar funções registradas aqui!
FUNCOES = {
    "consultar_carro": consultar_carro,
    "buscar_por_orcamento": buscar_por_orcamento,
    "aplicar_desconto": aplicar_desconto,
    "comparar_carros": comparar_carros,
}

# PROMPT DE SISTEMA: papel + anti-alucinação + GUARDRAIL DE ESCOPO
SISTEMA = {
    "role": "system",
    "content": (
        "Você é um consultor de vendas de carros. Responda sempre em português. "
        "Use as ferramentas para qualquer preço, busca, comparação ou cálculo — NUNCA invente valores. "
        "Os preços da base estão em dólares (USD). "
        "Se a ferramenta não encontrar o carro ou devolver um erro, explique isso educadamente ao cliente. "
        "GUARDRAIL DE ESCOPO: responda APENAS perguntas relacionadas a carros e à base de veículos; "
        "recuse educadamente qualquer assunto fora desse domínio."
    ),
}


# ==============================================================================
# PARTE D — LOOP ReAct COM OS 3 GUARDRAILS DO AGENTE
# Ciclo: Pensar (LLM) -> Agir (Python) -> Observar (role="tool") -> Repetir
# ==============================================================================
def agente(pergunta: str, max_iter: int = 5, verbose: bool = True) -> str:
    mensagens = [SISTEMA, {"role": "user", "content": pergunta}]

    for i in range(max_iter):  # GUARDRAIL 1: limite de passos (evita loop infinito)
        resp = client.chat.completions.create(
            model=MODELO,
            messages=mensagens,
            tools=ferramentas,
            tool_choice="auto",
            temperature=0.1,
        )
        msg = resp.choices[0].message

        # Sem tool_calls -> o LLM já formulou a resposta final!
        if not msg.tool_calls:
            return msg.content

        # Anexa o pedido do assistente ao histórico (model_dump evita problemas de serialização)
        mensagens.append(msg.model_dump(exclude_none=True))

        # Executa cada ferramenta solicitada
        for tc in msg.tool_calls:
            nome = tc.function.name
            args = {}
            func = FUNCOES.get(nome)  # GUARDRAIL 2: whitelist

            try:
                args = json.loads(tc.function.arguments or "{}")
                obs = func(**args) if func else {"erro": f"ferramenta '{nome}' não existe"}
            except Exception as e:  # GUARDRAIL 3: argumentos inválidos ou erro interno
                obs = {"erro": f"falha ao executar {nome}: {e}"}

            if verbose:
                print(f"  [{i + 1}] {nome}({args}) -> {obs}")

            # ⚠️ OBRIGATÓRIO: role="tool" + tool_call_id=tc.id + JSON string em content
            mensagens.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(obs, ensure_ascii=False, default=str),
                }
            )

    return "Não consegui concluir dentro do limite de passos."


if __name__ == "__main__":
    print("=== 1) Teste isolado das ferramentas (SEM LLM) ===")
    print("consultar_carro:", consultar_carro("bmw 320i"))
    print("buscar_por_orcamento:", buscar_por_orcamento(9000, "sedan"))
    print("aplicar_desconto OK:", aplicar_desconto(16430, 10))
    print("aplicar_desconto BLOQUEADO:", aplicar_desconto(16430, 90))
    print("comparar_carros:", comparar_carros("toyota corolla", "honda civic"))

    print("\n=== 2) Testes do Agente ReAct ===")
    perguntas = [
        "Tenho até 9 mil dólares e quero um sedan. Quais as opções e quanto fica a mais barata com 10% de desconto?",
        "Compare o toyota corolla com o honda civic.",
        "Me dá 90% de desconto no bmw 320i.",
        "Quanto custa um fusca?",
        "Qual a capital da França?",
    ]
    for p in perguntas:
        print(f"\n❓ {p}")
        print("🤖", agente(p))
