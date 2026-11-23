# Prova Walmir — Projeto de Referência (Vídeo)

Reconstrução do projeto exibido no vídeo de referência (`prova 1` / `Prova Walmir`), utilizando **LangGraph + `langgraph-supervisor` + Groq (`llama-3.3-70b-versatile`) + PyMuPDF (`fitz`) + ChromaDB**.

## Estrutura

- `agent/tools.py` — Ferramentas `@tool`: `buscar_na_web`, `resumir_texto` e `extrair_e_contar_pdf` (PyMuPDF/`fitz`).
- `agent/agent.py` — Agente Supervisor (`create_supervisor`) orquestrando `subagente_pesquisa` e `subagente_pln` sobre `agent/data/inteligencia artificial.pdf`.
- `agent/IA-UA09-Lab4-Python-Agente com langGraph - Agentes Versão 3.py` — Exemplo didático comentado com `buscar_na_web`, `resumir_texto`, `calcular` e supervisor multi-agente.
- `agent/load_data.py`, `agent/vector_store.py`, `agent/ingest.py`, `agent/retriever.py`, `agent/main.py` — Pipeline modular de RAG com ChromaDB persistente em `data/chroma/`.
- `agent/data/inteligencia artificial.pdf` — PDF usado nos testes (baseado em `introducaoml.pdf`).

## Como executar

No terminal, a partir da pasta `prova_video_walmir`:

```powershell
# 1. Rodar o Agente Supervisor (como no terminal do vídeo)
python agent/agent.py

# 2. Rodar o script didático Lab 4 (Versão 3)
python "agent/IA-UA09-Lab4-Python-Agente com langGraph - Agentes Versão 3.py"

# 3. Rodar o pipeline RAG modular com ChromaDB
python agent/main.py
```
