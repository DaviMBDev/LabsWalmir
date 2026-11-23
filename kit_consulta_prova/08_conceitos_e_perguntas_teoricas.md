# 08 — Conceitos e Respostas Prontas para Questões Teóricas / Arguição

Guia direto usando o vocabulário exato dos slides e labs do Prof. Walmir (UA01 a UA04 · Labs 00 a 09).

---

## 1. ML Clássico, EDA, Split e Overfitting (Labs 00–04)
- **`loc` vs `iloc`:** `iloc` seleciona por **posição inteira** (como lista Python, limite final exclusivo); `loc` seleciona por **rótulo/nome** ou **condição booleana** (limite final inclusivo).
- **Por que fazer EDA antes de modelar?** Para entender distribuições, detectar valores ausentes, corrigir *typos* (ex.: `maxda` → `mazda`), remover colunas irrelevantes (`car_ID`), identificar *outliers* e ver quais variáveis têm maior correlação com o alvo (`price`).
- **Train/Test Split (ex.: 80/20):** avalia se o modelo **generaliza** para dados que nunca viu. Nunca treinar nos dados de teste (*data leakage*).
- **Cross-Validation (K-Fold):** divide os dados em $K$ dobras (ex.: $K=5$), treina e testa $K$ vezes alternando a dobra de teste e tira a média — dá uma estimativa muito mais confiável do que um único split.
- **Métricas de Regressão:**
  - **MAE** (*Mean Absolute Error*): erro médio absoluto na mesma unidade do alvo (ex.: dólares).
  - **RMSE** (*Root Mean Squared Error*): penaliza mais erros grandes (sensível a *outliers*).
  - **$R^2$** (Coeficiente de Determinação): fração da variância explicada pelo modelo (mais próximo de `1.0` = melhor).
- **Overfitting vs. Underfitting:**
  - **Overfitting:** modelo excelente no treino ($R^2$ muito alto) e ruim no teste — **decorou** ruído em vez de aprender o padrão.
  - **Underfitting:** modelo ruim tanto no treino quanto no teste — simples demais para o padrão.

---

## 2. PLN Clássico (Lab 05 · UA03 Seq 02)
- **Pipeline clássico:** `ler PDF` (`pdfplumber` / `fitz`) → `normalizar` (minúsculas + regex) → `tokenizar` → `remover stop words` → `lematizar` (`spaCy`).
- **Stemming vs. Lematização:**
  - **Stemming (`RSLPStemmer`):** poda bruta que corta sufixos sem consultar gramática — gera radicais truncados (ex.: `"organizaç"`, `"complex"`). Mais rápido, porém impreciso em português.
  - **Lematização (`spaCy` `pt_core_news_sm`):** reduz à forma canônica/dicionário preservando palavras reais (ex.: `"organização"`, `"correr"`). Melhor escolha para português.
- **POS Tagging e NER:**
  - **POS Tagging:** classifica a classe gramatical de cada token (`NOUN`, `VERB`, `PROPN`, `ADJ`...).
  - **NER (*Named Entity Recognition*):** identifica entidades nomeadas no texto (`PER` pessoa, `ORG` organização, `LOC` local).
- **⚠️ PERGUNTA CLÁSSICA DE PROVA — Por que NÃO usar texto lematizado / sem stop words em Embeddings e RAG?**
  - Porque os modelos de **embeddings** e **LLMs** foram treinados em linguagem natural completa e usam **contexto** (atenção). Remover stop words (ex.: `"não"`, `"sem"`) e lematizar **destrói o significado semântico** da frase (ex.: `"o produto não funciona"` vira `"produto funcionar"`!). No RAG fazemos apenas **PLN leve** (extração de texto e limpeza de ruído de layout/quebras de linha).

---

## 3. Embeddings, Distâncias e Fundamentos de LLMs (Lab 06 · UA03 Seq 03–06)
- **Pirâmide de adaptação de GenAI** (da mais barata/simples à mais cara/complexa):
  1. **Prompt Engineering** (só instrução, sem treinar)
  2. **RAG** (injeta seus dados no contexto)
  3. **Fine-tuning** (reajusta pesos com exemplos)
  4. **Treinar do zero** (custo altíssimo)
- **Embedding:** representação vetorial (lista de números, ex.: 384 dimensões no `paraphrase-multilingual-MiniLM-L12-v2`) que captura o **significado semântico** do texto. Textos com sentidos parecidos ficam próximos no espaço vetorial mesmo sem compartilhar palavras (*"carro quebrou"* ≈ *"veículo com defeito"*).
- **Métricas de Distância / Similaridade:**
  - **Norma L1 (Manhattan):** soma das diferenças absolutas (como andar em quarteirões).
  - **Norma L2 (Euclidiana):** distância em linha reta entre dois pontos (padrão do ChromaDB — **menor = mais similar**).
  - **Similaridade de Cosseno:** mede o **ângulo/direção** entre dois vetores, ignorando magnitude (`1.0` = idênticos em direção, `0` = ortogonais, `-1` = opostos — **maior = mais similar**). Muito usada em PLN.
- **Mecanismo de Atenção & Transformer:** permite que o modelo olhe todas as palavras da frase em paralelo e entenda o contexto (ex.: `"banco"` de sentar vs. `"banco"` financeiro).
- **Tokens, Janela de Contexto e Temperatura:**
  - **Token:** pedaço de texto (subpalavra) que o modelo realmente processa; impacta custo e limite.
  - **Janela de Contexto:** orçamento máximo de tokens (instruções + histórico + documentos + resposta) que o modelo consegue "ver" de uma vez.
  - **Temperatura:** controla a aleatoriedade na escolha do próximo token — **baixa (`0.0`–`0.2`)** para RAG/Agentes (factual, determinístico); alta (`0.7`+) para criatividade.
- **Engenharia de Prompt:**
  - **Anatomia de um bom prompt:** **Papel** + **Tarefa** + **Contexto** + **Formato** de saída.
  - **Técnicas:** **Zero-shot** (só instrução), **Few-shot** (dá exemplos), **Chain-of-Thought** (pedir para "pensar passo a passo").

---

## 4. RAG, Chunking e ChromaDB (Labs 07–08 · UA03 Seq 07–08)
- **O que é RAG (*Retrieval-Augmented Generation*)?**
  - Recuperar (*Retrieval*) os trechos relevantes de uma base própria, injetá-los no prompt (*Augmented*) e gerar (*Generation*) a resposta fundamentada com um LLM — resolve o fato de o LLM não conhecer seus dados internos e reduz alucinações.
- **As 2 Fases do RAG:**
  1. **Fase 1 — Indexação (offline, feita 1 vez):** Ingestão do documento → Extração → Chunking → Embedding → Gravação no Vector DB (`col.add`).
  2. **Fase 2 — Consulta (online, a cada pergunta):** Pergunta → Embedding da pergunta → Busca vetorial top-k (`col.query`) → Montagem do prompt com contexto → Resposta do LLM.
- **Documento → Chunks → Tokens → Embedding:** *Chunking* você controla; *tokenização* ocorre dentro do modelo de embedding/LLM.
- **Estratégias de Chunking:**
  1. **Tamanho Fixo (`CharacterTextSplitter`):** corta rigidamente a cada N caracteres — simples, mas pode quebrar palavras/frases no meio.
  2. **Recursivo (`RecursiveCharacterTextSplitter` — PADRÃO):** tenta separadores naturais (`\n\n`, `\n`, `. `, ` `) em cascata — preserva parágrafos e frases.
  3. **Estrutural:** corta por seções/cabeçalhos (ex.: Markdown/HTML).
  4. **Semântico:** agrupa sentenças por mudança de significado (via embeddings).
  - **Para que serve o `chunk_overlap`?** Sobreposição entre chunks consecutivos para não perder informações que ficaram exatamente na fronteira do corte.
- **Banco de Dados Vetorial (ChromaDB):**
  - Diferente do SQL tradicional (busca exata `WHERE`), busca por **proximidade semântica** de vetores.
  - Cada item na `collection` guarda: **vetor (embedding) + ID + metadados (payload) + documento**.
  - Sempre verificar `if col.count() == 0:` antes de `col.add()` para evitar reindexar/duplicar a cada execução.

---

## 5. Grounding e Guardrails (Lab 08 · UA03 Seq 09)
- **Grounding (Fundamentação):** ancorar a resposta do modelo em dados reais (documentos, banco de dados, APIs), obrigando-o a responder **somente** com base no contexto recuperado, citar a fonte e admitir quando não sabe.
  - **Grounding verificável (numérico):** checar a distância retornada pelo ChromaDB (`scores[0] < LIMIAR`) — se nem o melhor chunk for próximo o suficiente, barra antes de chamar o LLM.
- **Guardrails ("trilhos de segurança"):** mecanismos de controle que limitam o que o sistema pode receber, fazer ou responder:
  1. **Guardrail de Entrada:** filtra perguntas vazias, curtas demais, maliciosas ou fora do domínio antes de gastar chamada ao LLM.
  2. **Guardrail de Grounding:** corta por limiar numérico quando não há evidência na base.
  3. **Guardrail de Escopo:** instruções no `SISTEMA` proibindo responder temas fora do domínio.
  4. **Guardrail de Negócio:** validação determinística em Python dentro da ferramenta (ex.: desconto só entre 0% e 30%).
  5. **Guardrail de Schema:** uso de `enum` e `required` no JSON Schema das ferramentas.
  6. **Guardrail de Saída:** checagem da resposta antes de entregar ao usuário.

---

## 6. Agentes, Function Calling, ReAct e Multi-Agentes (Lab 09 + Prova Passada)
- **Function Calling em 3 passos:**
  1. Você envia a pergunta + o catálogo JSON das ferramentas (`tools=...`) ao LLM.
  2. O LLM lê as descrições e devolve um pedido estruturado (`msg.tool_calls` com nome + argumentos em JSON) — **o LLM não executa código!**.
  3. O **seu código Python** executa a função real e devolve o resultado ao LLM como mensagem `{"role": "tool", "tool_call_id": tc.id, "content": json.dumps(obs)}`.
- **Loop ReAct (*Reason + Act*):**
  - Ciclo **Pensar → Agir → Observar** repetido em um `for` enquanto o modelo devolver `tool_calls`. Quando o modelo responde **sem** `tool_calls`, chegamos à resposta final. Permite encadear várias ferramentas em sequência sem programar a ordem fixa no código!
- **Os 3 Guardrails do Loop do Agente (Lab 09):**
  1. **Limite de passos (`max_iter`):** impede loop infinito / gasto excessivo de tokens.
  2. **Lista branca (`FUNCOES.get(nome)`):** impede execução arbitrária caso o LLM invente um nome de função.
  3. **Tratamento de exceções (`try/except`):** captura JSON malformado ou argumentos inválidos e devolve `{"erro": ...}` como observação para o agente se recuperar sem travar.
- **Agentic RAG:** transformar o pipeline de RAG (busca no ChromaDB) em **mais uma ferramenta** do agente — assim o próprio agente decide *quando* consultar o PDF, *quando* consultar o CSV e *quando* calcular.
- **Agente Único vs. Multi-Agente Supervisor (LangGraph — Prova Passada):**
  - Quando há muitos domínios/ferramentas, divide-se em **subagentes especialistas** (`create_react_agent`, ex.: `subagente_pesquisa`, `subagente_pln`, `subagente_calculo`) coordenados por um **Supervisor** (`create_supervisor`) que delega cada tarefa ao especialista adequado.
