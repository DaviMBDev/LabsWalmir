from langchain_core.tools import tool
import fitz  # PyMuPDF
import os


@tool
def buscar_na_web(query: str) -> str:
    """Busca informações textuais sobre um tema."""
    return f"Resultado para '{query}': Machine Learning são algoritmos que aprendem padrões para prever e decidir."


@tool
def resumir_texto(texto: str) -> str:
    """Resume um texto longo em poucas palavras."""
    return f"Resumo: {texto[:100]}..."


@tool
def extrair_e_contar_pdf(caminho_pdf: str) -> str:
    """Extrai texto de um PDF e conta o total de caracteres."""
    try:
        if not os.path.exists(caminho_pdf):
            return f"Erro: Arquivo não encontrado: {caminho_pdf}"

        doc = fitz.open(caminho_pdf)
        texto = "".join([pagina.get_text() for pagina in doc])
        doc.close()
        return f"PDF lido com sucesso. Total de caracteres: {len(texto)}"
    except Exception as e:
        return f"Erro ao processar o PDF: {str(e)}"
