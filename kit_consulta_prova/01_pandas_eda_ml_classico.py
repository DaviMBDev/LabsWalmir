"""
================================================================================
01 — PANDAS, FONTES DE DADOS, EDA E ML CLÁSSICO (Labs 00 a 04)
================================================================================
Guia de consulta rápida com receitas prontas para copiar e adaptar na prova.
Dataset de referência: CarPrice_Assignment.csv (na pasta raiz).
"""

import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================================================================
# 1. LEITURA DE FONTES DE DADOS (Lab 00 & Lab 02)
# ==============================================================================
def carregar_csv(caminho: str = "../CarPrice_Assignment.csv") -> pd.DataFrame:
    """Lê arquivo CSV local com separador vírgula."""
    return pd.read_csv(caminho, sep=",")


def ler_api_json(url: str) -> pd.DataFrame:
    """Exemplo Lab 02: consumir API REST retornando JSON e virar DataFrame."""
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def raspar_html_titulos(url: str) -> list[str]:
    """Exemplo Lab 02: Web Scraping simples com BeautifulSoup."""
    resp = requests.get(url, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    return [h.get_text(strip=True) for h in soup.find_all(["h1", "h2"])]


# ==============================================================================
# 2. SELEÇÃO EM PANDAS: loc vs iloc (Lab 00)
# ==============================================================================
# Regra prática:
#   - iloc -> seleção POSICIONAL (por índice inteiro, igual lista Python; fim EXCLUSIVO)
#   - loc  -> seleção por RÓTULO / CONDIÇÃO booleana (fim INCLUSIVO em slices)
def exemplos_selecao_pandas(df: pd.DataFrame):
    primeiras_5_linhas = df.iloc[:5, :3]                    # 5 linhas, 3 colunas por posição
    sedans_baratos = df.loc[
        (df["carbody"] == "sedan") & (df["price"] <= 9000),
        ["CarName", "carbody", "horsepower", "price"],
    ]
    ordenado = sedans_baratos.sort_values("price").head(5)
    return primeiras_5_linhas, ordenado


# ==============================================================================
# 3. EDA, LIMPEZA DE TYPOS E CORRELAÇÃO (Lab 03)
# ==============================================================================
def limpar_e_analisar_carros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Receita clássica do Lab 03:
    1) Padronizar texto (lower + strip)
    2) Extrair marca (primeira palavra de CarName)
    3) Corrigir erros de digitação (typos) com .replace()
    """
    df = df.copy()
    df["modelo"] = df["CarName"].str.lower().str.strip()
    df["marca"] = df["modelo"].str.split().str[0]

    correcoes = {
        "maxda": "mazda",
        "porcshce": "porsche",
        "toyouta": "toyota",
        "vokswagen": "volkswagen",
        "vw": "volkswagen",
    }
    df["marca"] = df["marca"].replace(correcoes)
    return df


def inspecao_rapida_eda(df: pd.DataFrame, alvo: str = "price"):
    """Retorna diagnóstico básico: shape, nulos, duplicados e top correlações."""
    correlacoes = (
        df.select_dtypes(include=[np.number])
        .drop(columns=["car_ID"], errors="ignore")
        .corr()[alvo]
        .drop(labels=[alvo], errors="ignore")
        .reindex(
            df.select_dtypes(include=[np.number])
            .drop(columns=["car_ID", alvo], errors="ignore")
            .corrwith(df[alvo])
            .abs()
            .sort_values(ascending=False)
            .index
        )
    )
    return {
        "shape": df.shape,
        "nulos_total": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "top_5_correlacoes": correlacoes.head(5).round(3).to_dict(),
    }


# ==============================================================================
# 4. TREINAMENTO, TESTES, K-FOLD E OVERFITTING CHECK (Lab 04)
# ==============================================================================
def treinar_e_avaliar_regressao(
    df: pd.DataFrame,
    features: list[str] = None,
    alvo: str = "price",
    test_size: float = 0.2,
    k_folds: int = 5,
) -> dict:
    """
    Pipeline completo do Lab 04:
    - train_test_split (80/20)
    - LinearRegression()
    - Métricas MAE, RMSE e R² em TREINO e TESTE (diagnóstico de overfitting)
    - Validação cruzada K-Fold (cross_val_score)
    """
    if features is None:
        features = ["enginesize", "horsepower", "curbweight", "highwaympg"]

    X = df[features]
    y = df[alvo]

    # 1. Split Treino / Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # 2. Treinar SOMENTE no treino (evita data leakage!)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # 3. Previsões e métricas
    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    metricas = {
        "treino": {
            "MAE": round(float(mean_absolute_error(y_train, pred_train)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_train, pred_train))), 2),
            "R2": round(float(r2_score(y_train, pred_train)), 4),
        },
        "teste": {
            "MAE": round(float(mean_absolute_error(y_test, pred_test)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred_test))), 2),
            "R2": round(float(r2_score(y_test, pred_test)), 4),
        },
    }

    # 4. Cross-Validation (K-Fold)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    scores_cv = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
    metricas["kfold"] = {
        "k": k_folds,
        "scores_r2": [round(float(s), 4) for s in scores_cv],
        "r2_medio": round(float(scores_cv.mean()), 4),
        "r2_desvio": round(float(scores_cv.std()), 4),
    }

    # Regra de diagnóstico de overfitting:
    # - Se R² treino >> R² teste (diferença grande, ex. 0.95 vs 0.55) -> OVERFITTING (decorou o treino)
    # - Se ambos forem baixos -> UNDERFITTING (modelo simples demais)
    # - Se ambos forem altos e próximos -> BOA GENERALIZAÇÃO
    return metricas


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "CarPrice_Assignment.csv")
    df = limpar_e_analisar_carros(carregar_csv(csv_path))
    print("EDA:", inspecao_rapida_eda(df))
    print("ML:", treinar_e_avaliar_regressao(df))
