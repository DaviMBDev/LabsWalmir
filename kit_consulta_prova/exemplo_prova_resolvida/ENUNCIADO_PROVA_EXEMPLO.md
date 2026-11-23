# 📝 PROVA PRÁTICA MODELO — INTELIGÊNCIA ARTIFICIAL (EXEMPLO RESOLVIDO)
**Estilo: Prof. Walmir (FATEC) · Baseada na Prova Passada + Labs 05 a 09**
**Valor total:** 10,0 pontos

---

## Contexto da Prova
Você foi contratado para construir um **Assistente Inteligente Híbrido** capaz de atender usuários sobre **dois domínios** usando os arquivos fornecidos em `data/`:
1. **Base Estruturada (`data/CarPrice_Assignment.csv`)**: catálogo de veículos (preços em USD, carrocerias, combustível, potência).
2. **Base Não Estruturada (`data/introducaoml.pdf`)**: apostila técnica de Machine Learning.

O sistema precisa ser **seguro (com Guardrails)**, **fundamentado (com Grounding)** e capaz de **decidir sozinho** qual ferramenta ou subagente acionar para resolver perguntas simples e compostas.

---

## Requisitos da Prova

### Requisito 1 — Ingestão de PDF, Chunking Recursivo, ChromaDB e Guardrails (2,5 pts)
Implementado em `rag_chroma.py`:
- Extrair o texto **natural** de `data/introducaoml.pdf` (sem lematização nem remoção de stopwords).
- Dividir em chunks com `RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=60)`.
- Indexar em um banco vetorial **ChromaDB persistente** (`./chroma_prova`) usando `paraphrase-multilingual-MiniLM-L12-v2`, protegendo contra duplicação (`if col.count() == 0`).
- Implementar **Guardrail de Entrada** (bloqueia perguntas vazias ou `< 5` caracteres) e **Guardrail de Grounding Numérico** (`scores[0] < LIMIAR`).

### Requisito 2 — Limpeza de Dados e Ferramentas Python Testadas Isoladamente (2,0 pts)
Implementado em `tools_carros.py`:
- Limpar *typos* nas marcas de `CarPrice_Assignment.csv` (`maxda`, `porcshce`, `toyouta`, `vokswagen`, `vw`).
- Implementar 6 ferramentas Python retornando sempre `dict` (inclusive em caso de erro):
  1. `consultar_carro(modelo)` — preço médio, carroceria, combustível e potência média.
  2. `buscar_por_orcamento(preco_max, carroceria, limite)` — carros mais baratos até o orçamento.
  3. `aplicar_desconto(preco, percentual)` — **guardrail de negócio**: aceita apenas `0%` a `30%`.
  4. `comparar_carros(modelo_a, modelo_b)` — compara preço médio, diferença, mais barato e mais potente.
  5. `analisar_pdf(caminho_pdf)` — lê PDF com PyMuPDF (`fitz`), valida existência e conta páginas, palavras e caracteres (estilo prova passada!).
  6. `consultar_apostila_ml(pergunta)` — **Agentic RAG**: consulta o ChromaDB com guardrail de distância.

### Requisito 3 — Agente ReAct em Groq Puro com 3 Guardrails (3,0 pts)
Implementado em `agente_react_prova.py`:
- Descrever todas as 6 ferramentas no catálogo JSON `ferramentas` (usando `enum` em `carroceria`).
- Criar whitelist `FUNCOES` e prompt `SISTEMA` com **guardrail de escopo** (recusar assuntos fora de carros e ML).
- Implementar o loop ReAct `agente(pergunta, max_iter=5, verbose=True)` com:
  - Guardrail 1: limite de iterações (`max_iter`).
  - Guardrail 2: lista branca (`FUNCOES.get(nome)`).
  - Guardrail 3: tratamento de exceções (`try/except`).
- Executar a bateria completa de 8 cenários (incluindo 2 ferramentas em sequência, desconto abusivo, carro inexistente, fora de escopo, análise de PDF, Agentic RAG e teste com `max_iter=1`).

### Requisito 4 — Versão Multi-Agente Supervisor com LangGraph (1,5 pt)
Implementado em `agente_supervisor_prova.py`:
- Reestruturar as ferramentas com `@tool` em 2 subagentes especialistas (`subagente_carros` e `subagente_pln_rag`) usando `create_react_agent`.
- Orquestrar ambos com `create_supervisor` (`langgraph-supervisor`) e demonstrar a delegação em perguntas sobre PDF, apostila (RAG) e carros.

### Requisito 5 — Relatório de Resultados e Justificativas Técnicas (1,0 pt)
Entregue em `RELATORIO_RESULTADOS.md` explicando as decisões técnicas e os resultados dos guardrails.
