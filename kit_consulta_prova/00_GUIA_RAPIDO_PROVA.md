# 🎯 00 — GUIA RÁPIDO PARA A PROVA DE IA (PROF. WALMIR)

Mapa mestre da pasta `kit_consulta_prova/` para encontrar qualquer código ou conceito em segundos durante a prova.

---

## 🗺️ Se a prova pedir X → Abra o arquivo Y

| Se o enunciado da prova pedir... | Abra este arquivo |
|---|---|
| Ler CSV/API/HTML, limpar *typos*, `loc`/`iloc`, correlação, Train/Test Split, Regressão Linear, K-Fold, MAE/RMSE/R² | [01_pandas_eda_ml_classico.py](file:///c:/Users/davim/OneDrive/Documentos/FATEC/estudar%20Walmir/kit_consulta_prova/01_pandas_eda_ml_classico.py) |
| Ler PDF (`pdfplumber` ou `fitz`/PyMuPDF), contar caracteres/palavras/páginas, PLN clássico (`spaCy`, `NLTK`, lematização vs stemming, POS/NER) | [02_pln_classico_e_pdf.py](file:///c:/Users/davim/OneDrive/Documentos/FATEC/estudar%20Walmir/kit_consulta_prova/02_pln_classico_e_pdf.py) |
| Embeddings (`SentenceTransformer`), `cosine_similarity`, busca semântica manual, Chunking Fixo vs Recursivo, RAG manual com Groq | [03_embeddings_chunking_rag.py](file:///c:/Users/davim/OneDrive/Documentos/FATEC/estudar%20Walmir/kit_consulta_prova/03_embeddings_chunking_rag.py) |
| Banco vetorial **ChromaDB** persistente, `col.add`/`col.query`, Guardrail de Entrada, Guardrail de Grounding numérico (`LIMIAR`) | [04_rag_chromadb_guardrails.py](file:///c:/Users/davim/OneDrive/Documentos/FATEC/estudar%20Walmir/kit_consulta_prova/04_rag_chromadb_guardrails.py) |
| **Agente em Groq puro (Lab 09)**: ferramentas Python, Schema JSON (`tools`), `FUNCOES` whitelist, `SISTEMA`, **loop ReAct `agente()`** com guardrails | [05_agente_react_groq_puro.py](file:///c:/Users/davim/OneDrive/Documentos/FATEC/estudar%20Walmir/kit_consulta_prova/05_agente_react_groq_puro.py) |
| **Agente Supervisor com LangGraph (igual ao vídeo da prova passada)**: `@tool`, `create_react_agent`, `create_supervisor`, `fitz` | [06_agente_supervisor_langgraph.py](file:///c:/Users/davim/OneDrive/Documentos/FATEC/estudar%20Walmir/kit_consulta_prova/06_agente_supervisor_langgraph.py) |
| **Template CORINGA (Agentic RAG completo)**: une CSV + PDF + ChromaDB + Agente ReAct + todos os guardrails num lugar só | [07_agentic_rag_hibrido_pronto.py](file:///c:/Users/davim/OneDrive/Documentos/FATEC/estudar%20Walmir/kit_consulta_prova/07_agentic_rag_hibrido_pronto.py) |
| Responder perguntas **teóricas / discursivas** ou explicar ao professor na arguição | [08_conceitos_e_perguntas_teoricas.md](file:///c:/Users/davim/OneDrive/Documentos/FATEC/estudar%20Walmir/kit_consulta_prova/08_conceitos_e_perguntas_teoricas.md) |
| Ver uma **prova completa resolvida de ponta a ponta** (enunciado + RAG + 6 tools + ReAct + LangGraph Supervisor + relatório) | [ENUNCIADO_PROVA_EXEMPLO.md](file:///c:/Users/davim/OneDrive/Documentos/FATEC/estudar%20Walmir/kit_consulta_prova/exemplo_prova_resolvida/ENUNCIADO_PROVA_EXEMPLO.md) |

---

## 🚨 Comandos Salva-Vidas na Hora da Prova

### 1. Modelo da Groq que funciona na sua conta (NÃO use `llama-3.3-70b-versatile` — deu 404!)
- **Principal recomendado (suporta tool calling e supervisor):** `openai/gpt-oss-120b`
- **Alternativas ativas na sua conta:** `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`, `qwen/qwen3.8-27b`

Se algum modelo falhar na hora da prova, rode isso para listar os modelos ativos:
```python
import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
c = Groq(api_key=os.getenv("GROQ_API_KEY"))
print(sorted(m.id for m in c.models.list().data))
```

### 2. Modelo de Embedding local já em cache no seu PC (funciona até offline!)
```python
"paraphrase-multilingual-MiniLM-L12-v2"   # 384 dimensões, excelente em português
```

### 3. Snippet de conexão segura com `.env`
```python
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
GROQ_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_2")
```

---

## ⚠️ As 8 Pegadinhas Clássicas que Derrubam na Prova (Checklist Anti-Erro)

1. **Texto natural para Embeddings/RAG:** NUNCA passe o texto lematizado/sem stopwords do Lab 05 para `SentenceTransformer` ou `ChromaDB`.
2. **Sinal da distância no ChromaDB:** `col.query()` devolve **distâncias** → **MENOR = MAIS SIMILAR** (`scores[0] < LIMIAR`). Já `cosine_similarity()` devolve **similaridade** → **MAIOR = MAIS SIMILAR**.
3. **Duplicação no ChromaDB:** sempre proteja a indexação com `if col.count() == 0:` antes de `col.add(...)`.
4. **Testar ferramentas SEM LLM primeiro:** rode `print(minha_ferramenta(...))` antes de plugar no agente. Sempre cheque `os.path.exists(...)` para PDFs/CSVs e envolva em `try/except`.
5. **Retorno das ferramentas no Lab 09:** cada ferramenta deve retornar um `dict` simples (inclusive em caso de erro: `{"erro": "..."}`) e a mensagem de resposta no loop ReAct precisa ter `"role": "tool"` e `"tool_call_id": tc.id`.
6. **Erro ao anexar `msg` no histórico da Groq:** se der erro de validação na segunda iteração do ReAct, use `mensagens.append(msg.model_dump(exclude_none=True))`.
7. **A `description` / docstring é prompt:** se o agente não chama sua ferramenta ou passa o argumento errado, melhore a docstring/description e use `"enum": [...]` para valores fixos.
8. **Temperatura baixa em RAG e Agentes:** use `temperature=0` ou `0.1`–`0.2` para evitar alucinações e erros de JSON em tool calls.

---

## ✅ Checklist Rápido (5 minutos antes de abrir a prova amanhã)
- [ ] Conferir se o arquivo `.env` está na pasta onde você vai rodar o código.
- [ ] Conferir que o interpretador Python selecionado no VS Code é o **Python 3.10** (onde já instalamos tudo: `pandas`, `sklearn`, `pdfplumber`, `fitz`/`pymupdf`, `spacy`, `nltk`, `sentence_transformers`, `chromadb`, `groq`, `langchain_groq`, `langgraph`, `langgraph_supervisor`).
- [ ] Deixar esta pasta `kit_consulta_prova/` aberta no VS Code em outra aba para consulta rápida.
