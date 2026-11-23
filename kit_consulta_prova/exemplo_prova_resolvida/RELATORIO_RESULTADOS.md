# 📊 Relatório de Resultados e Justificativas Técnicas (Requisito 5)

Modelo pronto de como documentar/responder as perguntas discursivas que acompanham a prova prática.

---

## 1. Por que o texto extraído do PDF para o ChromaDB NÃO passou por lematização nem remoção de stopwords?
Porque modelos de *embeddings* (`paraphrase-multilingual-MiniLM-L12-v2`) e LLMs dependem do **contexto sintático e semântico completo** da frase. O pré-processamento agressivo do PLN clássico (Lab 05) remove palavras funcionais importantes (como `"não"`, `"sem"`, preposições) e trunca flexões, degradando a qualidade dos vetores e da recuperação no RAG. Para RAG, aplicamos apenas **PLN leve** (extração e limpeza de layout).

## 2. Por que usamos `RecursiveCharacterTextSplitter` com `chunk_overlap`?
- O corte **recursivo** respeita fronteiras naturais (`\n\n`, `\n`, `. `, espaço), evitando partir palavras ou frases ao meio como ocorre no corte por tamanho fixo (`CharacterTextSplitter`).
- O **`chunk_overlap`** (60 caracteres) mantém uma sobreposição entre chunks vizinhos para que informações localizadas exatamente na borda do corte não percam contexto.

## 3. Como funcionam os Guardrails implementados?
- **Guardrail de Entrada:** barra perguntas vazias ou com menos de 5 caracteres antes de consumir consultas ao banco vetorial ou ao LLM.
- **Guardrail de Grounding Numérico (ChromaDB):** verifica `scores[0] < 1.25` (no ChromaDB, **menor distância = maior similaridade**). Se nem o melhor chunk for próximo o suficiente, recusa responder em vez de deixar o LLM alucinar.
- **Guardrail de Negócio (`aplicar_desconto`):** valida deterministicamente em Python `0 <= percentual <= 30`. Ao pedir 80%, a ferramenta retorna `{"erro": "desconto permitido: de 0% a 30%"}` e o LLM comunica a recusa sem inventar cálculo.
- **Guardrail de Schema (`enum` em `carroceria`):** restringe os valores que o modelo pode enviar na chamada JSON.
- **Guardrail de Escopo (`SISTEMA`):** instrui o modelo a recusar temas fora de carros e Machine Learning (ex.: *"Qual a capital da França?"*).
- **3 Guardrails do Loop ReAct (`agente()`):**
  1. `max_iter`: impede loop infinito / gasto excessivo de tokens.
  2. `FUNCOES.get(nome)` (whitelist): impede chamar funções inexistentes ou não autorizadas.
  3. `try/except`: trata JSON malformado ou argumentos inválidos sem quebrar a aplicação.

## 4. O que acontece no teste com `max_iter=1` para uma pergunta que exige 2 ferramentas?
Na iteração 1, o modelo chama a primeira ferramenta (`buscar_por_orcamento`), recebe a observação no histórico, e o laço `for` se encerra antes que o modelo possa executar a segunda chamada (`aplicar_desconto`) e formular a resposta final. Por isso, o guardrail de limite de passos entra em ação e retorna a mensagem segura: `"Não consegui concluir dentro do limite de passos."`

## 5. Por que essa arquitetura caracteriza um **Agentic RAG**?
No RAG tradicional (Lab 08), toda pergunta passa obrigatoriamente pela busca vetorial no PDF. Ao transformar `consultar_apostila_ml` em uma **ferramenta do agente** (junto com ferramentas de CSV e análise de PDF), o próprio LLM decide autonomamente *quando* consultar o ChromaDB, *quando* consultar o CSV e *quando* combinar ferramentas em sequência.

---

## 6. Evidências de Execução Real (`agente_react_prova.py`)
- **Multi-step ReAct (2 ferramentas):** `buscar_por_orcamento(9000, 'sedan')` retornou **Nissan Versa (US\$ 5.499)** como mais barato → `aplicar_desconto(5499, 15)` retornou **US\$ 4.674,15**.
- **Comparação:** `comparar_carros('toyota corolla', 'honda civic')` → Corolla (**US\$ 9.409,83 · 89 hp**) vs. Civic (**US\$ 8.192,75 · 79 hp**), diferença de **US\$ 1.217,08**.
- **Guardrail de Negócio (80%):** recusou informando o limite máximo de 30%.
- **Carro Inexistente (`fusca`):** ferramenta devolveu `{"erro": "modelo 'fusca' não encontrado na base"}` e o agente admitiu sem inventar preço.
- **Guardrail de Escopo (`capital da França`):** recusou responder fora do domínio.
- **Análise de PDF (`analisar_pdf`):** `introducaoml.pdf` → **1 página, 253 palavras, 1.862 caracteres**.
- **Agentic RAG (`consultar_apostila_ml`):** recuperou os trechos da apostila com `menor_distancia = 0.132` (`< 1.25`) e respondeu fundamentado.
- **Guardrail de Passos (`max_iter=1`):** encerrou após o passo 1 retornando `"Não consegui concluir dentro do limite de passos."`
