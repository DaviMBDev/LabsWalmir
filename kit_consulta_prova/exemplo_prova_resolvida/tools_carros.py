"""
REQUISITO 2 — Limpeza de Dados (Lab 03) + 6 Ferramentas Python Testáveis Isoladamente (Lab 09 + Prova Passada)
"""

import os
import sys
import pandas as pd
import fitz  # PyMuPDF (usado na prova passada)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rag_chroma import buscar_na_apostila

CSV_PADRAO = os.path.join(BASE_DIR, "data", "CarPrice_Assignment.csv")
PDF_PADRAO = os.path.join(BASE_DIR, "data", "introducaoml.pdf")


def carregar_base_carros() -> pd.DataFrame:
    """Carrega o CSV, padroniza modelo e corrige os typos clássicos de marca (Lab 03)."""
    df = pd.read_csv(CSV_PADRAO)
    df["modelo"] = df["CarName"].str.lower().str.strip()
    df["marca"] = df["modelo"].str.split().str[0].replace(
        {
            "maxda": "mazda",
            "porcshce": "porsche",
            "toyouta": "toyota",
            "vokswagen": "volkswagen",
            "vw": "volkswagen",
        }
    )
    return df


df_carros = carregar_base_carros()


# ==============================================================================
# AS 6 FERRAMENTAS PYTHON (TODAS RETORNAM DICT, INCLUSIVE EM ERRO!)
# ==============================================================================
def consultar_carro(modelo: str) -> dict:
    """1) Consulta preço médio e características de um modelo no CSV."""
    achados = df_carros[df_carros["modelo"].str.contains(modelo.lower().strip(), regex=False)]
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
    """2) Lista os carros mais baratos até preco_max, filtrando opcionalmente por carroceria."""
    base = df_carros[df_carros["price"] <= float(preco_max)]
    if carroceria:
        base = base[base["carbody"] == carroceria.lower().strip()]
    base = base.sort_values("price").drop_duplicates("modelo").head(int(limite))
    if base.empty:
        return {"erro": "nenhum carro encontrado com esses critérios"}
    return {"opcoes": base[["modelo", "carbody", "horsepower", "price"]].to_dict("records")}


def aplicar_desconto(preco: float, percentual: float) -> dict:
    """3) Aplica desconto com GUARDRAIL DE NEGÓCIO (somente 0% a 30%)."""
    if not 0 <= float(percentual) <= 30:
        return {"erro": "desconto permitido: de 0% a 30%"}
    final = float(preco) * (1 - float(percentual) / 100)
    return {
        "preco_original": float(preco),
        "percentual": float(percentual),
        "preco_final": round(final, 2),
    }


def comparar_carros(modelo_a: str, modelo_b: str) -> dict:
    """4) Compara dois modelos lado a lado (Desafio extra do Lab 09)."""
    a = consultar_carro(modelo_a)
    if "erro" in a:
        return {"erro": f"Primeiro modelo inválido: {a['erro']}"}
    b = consultar_carro(modelo_b)
    if "erro" in b:
        return {"erro": f"Segundo modelo inválido: {b['erro']}"}

    return {
        "modelo_a": {"nome": modelo_a, "preco_medio_usd": a["preco_medio_usd"], "potencia_media_hp": a["potencia_media_hp"]},
        "modelo_b": {"nome": modelo_b, "preco_medio_usd": b["preco_medio_usd"], "potencia_media_hp": b["potencia_media_hp"]},
        "diferenca_preco_usd": round(abs(a["preco_medio_usd"] - b["preco_medio_usd"]), 2),
        "mais_barato": modelo_a if a["preco_medio_usd"] <= b["preco_medio_usd"] else modelo_b,
        "mais_potente": modelo_a if a["potencia_media_hp"] >= b["potencia_media_hp"] else modelo_b,
    }


def analisar_pdf(caminho_pdf: str = None) -> dict:
    """5) Extrai texto de um PDF com PyMuPDF e conta páginas, palavras e caracteres (estilo prova passada)."""
    caminho = caminho_pdf if caminho_pdf and os.path.exists(caminho_pdf) else PDF_PADRAO
    try:
        if not os.path.exists(caminho):
            return {"erro": f"Arquivo não encontrado: {caminho}"}
        doc = fitz.open(caminho)
        paginas = len(doc)
        texto = "".join(p.get_text() for p in doc)
        doc.close()
        return {
            "arquivo": os.path.basename(caminho),
            "paginas": paginas,
            "total_palavras": len(texto.split()),
            "total_caracteres": len(texto),
        }
    except Exception as e:
        return {"erro": f"Erro ao processar PDF: {str(e)}"}


def consultar_apostila_ml(pergunta: str) -> dict:
    """6) Agentic RAG: busca na apostila de Machine Learning via ChromaDB protegido."""
    return buscar_na_apostila(pergunta, k=2)


if __name__ == "__main__":
    print("=== VALIDAÇÃO ISOLADA DAS 6 FERRAMENTAS (SEM LLM) ===")
    print("1. consultar_carro('bmw 320i'):", consultar_carro("bmw 320i"))
    print("1b. consultar_carro('fusca'):", consultar_carro("fusca"))
    print("2. buscar_por_orcamento(9000, 'sedan'):", buscar_por_orcamento(9000, "sedan"))
    print("3a. aplicar_desconto(16430, 10):", aplicar_desconto(16430, 10))
    print("3b. aplicar_desconto(16430, 90):", aplicar_desconto(16430, 90))
    print("4. comparar_carros:", comparar_carros("toyota corolla", "honda civic"))
    print("5. analisar_pdf:", analisar_pdf(PDF_PADRAO))
    print("6. consultar_apostila_ml:", consultar_apostila_ml("O que é aprendizado supervisionado?"))
