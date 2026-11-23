import os
import sys
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

load_dotenv(os.path.join(PROJECT_DIR, ".env"))

from groq import Groq
from ingest import executar_ingestao
from retriever import buscar_contexto

LIMIAR_DISTANCIA = 1.25


def responder_rag(pergunta: str) -> str:
    if len(pergunta.strip()) < 5:
        return "Pergunta muito curta ou vazia."

    docs, dists = buscar_contexto(pergunta, k=3)
    if not dists or dists[0] > LIMIAR_DISTANCIA:
        return "Não encontrei essa informação na base de conhecimento."

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    contexto = "\n\n".join(docs)
    prompt = (
        "Responda em português usando APENAS o contexto abaixo. "
        "Se a resposta não estiver no contexto, diga que não encontrou.\n\n"
        f"CONTEXTO:\n{contexto}\n\nPERGUNTA: {pergunta}"
    )
    resp = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    executar_ingestao()
    for q in [
        "O que é aprendizado supervisionado?",
        "Qual a receita de lasanha?",
    ]:
        print(f"\n❓ {q}")
        print("🤖", responder_rag(q))
