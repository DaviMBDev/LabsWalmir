"""
================================================================================
02 — PLN CLÁSSICO E LEITURA DE PDF (Lab 05 + Prova Passada)
================================================================================
Contém:
  1) Extração de texto de PDF com pdfplumber (Labs 05-08) E PyMuPDF/fitz (Prova Passada)
  2) Estatísticas rápidas de PDF (páginas, caracteres, palavras)
  3) Pipeline de PLN Clássico (minúsculas, regex, spaCy lematização, NLTK stopwords)
  4) Comparativo prático: Stemming (RSLPStemmer) vs Lematização (spaCy)
  5) Análise gramatical e semântica: POS Tagging e NER (spaCy)

⚠️ PEGADINHA CLÁSSICA DA PROVA:
   Use `processar_texto_pln()` APENAS para PLN clássico / estatística de termos.
   Para EMBEDDINGS e RAG (Labs 06-08), SEMPRE use o TEXTO NATURAL do PDF!
   Remover stopwords e lematizar destrói o contexto que os modelos de embedding precisam.
"""

import os
import re
import pdfplumber
import fitz  # PyMuPDF (usado na prova do semestre passado)
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer


# ==============================================================================
# 1. LEITURA DE PDF — OPÇÃO A: pdfplumber (padrão dos Labs 05 a 08)
# ==============================================================================
def ler_pdf_pdfplumber(caminho_pdf: str) -> str:
    """Extrai todo o texto natural de um PDF usando pdfplumber."""
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"PDF não encontrado: {caminho_pdf}")
    texto = ""
    with pdfplumber.open(caminho_pdf) as pdf:
        for pagina in pdf.pages:
            t = pagina.extract_text()
            if t:
                texto += t + " "
    return texto.replace("\n", " ").strip()


# ==============================================================================
# 2. LEITURA DE PDF — OPÇÃO B: PyMuPDF / fitz (usado na prova passada!)
# ==============================================================================
def ler_pdf_pymupdf(caminho_pdf: str) -> dict:
    """Extrai texto de um PDF com fitz (PyMuPDF) e já devolve estatísticas úteis."""
    if not os.path.exists(caminho_pdf):
        return {"erro": f"Arquivo não encontrado: {caminho_pdf}"}
    try:
        doc = fitz.open(caminho_pdf)
        paginas = len(doc)
        texto = "".join(pagina.get_text() for pagina in doc)
        doc.close()
        return {
            "caminho": caminho_pdf,
            "paginas": paginas,
            "total_caracteres": len(texto),
            "total_palavras": len(texto.split()),
            "texto": texto,
        }
    except Exception as e:
        return {"erro": f"Erro ao processar PDF: {str(e)}"}


# ==============================================================================
# 3. PIPELINE COMPLETO DE PLN CLÁSSICO (Lab 05)
# ==============================================================================
def carregar_recursos_pln():
    """Carrega modelo spaCy PT-BR e conjunto de stopwords (NLTK + customizadas)."""
    try:
        api_sw = set(stopwords.words("portuguese"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        api_sw = set(stopwords.words("portuguese"))

    minhas_sw = {"a", "e", "i", "o", "u", "neste", "nesta"}
    stop_words = api_sw | minhas_sw
    nlp = spacy.load("pt_core_news_sm")
    return nlp, stop_words


def processar_texto_pln(texto: str, nlp=None, stop_words=None) -> str:
    """
    Pipeline clássico:
    1) Minúsculas
    2) Regex mantendo apenas letras (PT-BR) e espaços
    3) Tokenização + Lematização com spaCy
    4) Remoção de stopwords, pontuação e espaços vazios
    """
    if nlp is None or stop_words is None:
        nlp, stop_words = carregar_recursos_pln()

    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúâêîôûãõç\s]", "", texto)
    doc = nlp(texto)
    clean_tokens = [
        token.lemma_
        for token in doc
        if token.text not in stop_words
        and not token.is_punct
        and token.text.strip() != ""
    ]
    return " ".join(clean_tokens)


# ==============================================================================
# 4. STEMMING vs LEMATIZAÇÃO (Lab 05)
# ==============================================================================
def comparar_stemming_lematizacao(frase: str, nlp=None) -> dict:
    """
    Stemming (poda bruta): corta radicais -> pode gerar palavras truncadas ('organizaç').
    Lematização (canônica): consulta gramática/dicionário -> preserva palavras reais ('organização', 'correr').
    Em português, lematização costuma ser a melhor escolha.
    """
    if nlp is None:
        nlp = spacy.load("pt_core_news_sm")
    try:
        stemmer = RSLPStemmer()
    except LookupError:
        nltk.download("rslp", quiet=True)
        stemmer = RSLPStemmer()

    tokens = re.findall(r"[A-Za-zÀ-ÿ]+", frase)
    stems = [stemmer.stem(t) for t in tokens]
    lemas = [t.lemma_ for t in nlp(frase) if not t.is_punct]
    return {
        "original": frase,
        "stemming": " ".join(stems),
        "lematizacao": " ".join(lemas),
    }


# ==============================================================================
# 5. POS TAGGING E NER COM SPACY (Lab 05)
# ==============================================================================
def analisar_pos_e_ner(frase: str, nlp=None) -> dict:
    """
    POS Tagging: classe gramatical de cada token (PROPN, VERB, NOUN, ADJ...).
    NER (Named Entity Recognition): entidades nomeadas (PER, ORG, LOC, MISC).
    """
    if nlp is None:
        nlp = spacy.load("pt_core_news_sm")
    doc = nlp(frase)
    return {
        "pos": [(t.text, t.pos_, t.lemma_) for t in doc],
        "ner": [(ent.text, ent.label_) for ent in doc.ents],
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, "introducaoml.pdf")
    info = ler_pdf_pymupdf(pdf_path)
    print(f"PDF: {info['paginas']} pág | {info['total_caracteres']} chars | {info['total_palavras']} palavras")

    nlp, sw = carregar_recursos_pln()
    tratado = processar_texto_pln(info["texto"], nlp, sw)
    print(f"Texto tratado PLN: {len(tratado)} chars -> {tratado[:120]}...")
    print("Stemming vs Lema:", comparar_stemming_lematizacao("As organizações correram rapidamente.", nlp))
    print("POS/NER:", analisar_pos_e_ner("Steve Jobs fundou a Apple em 1976 na cidade de Cupertino.", nlp))
