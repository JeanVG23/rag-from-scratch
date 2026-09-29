# Comparaison LlamaIndex vs RAG from-scratch

Évaluation générée le 2026-09-29 20:36 UTC.
- Framework comparé : LlamaIndex end-to-end (`SimpleDirectoryReader` + `SentenceSplitter` + `VectorStoreIndex`).
- Modèle d'embeddings : `bge-m3` via Ollama.
- Modèle de génération : `qwen3.5:4b` via Ollama (température 0, limite 384 tokens).
- Jeu de questions : `evaluation/private/ia04_questions_dev.jsonl` (15 questions, dont 10 répondables et 5 sans réponse).
- Découpage LlamaIndex : `chunk_size=512`, `chunk_overlap=50` $\rightarrow$ **383 chunks** produits à partir des 20 documents IA04.

## 1. Métriques de Recherche (Retrieval sur les questions répondables)

| Système | Type d'index | Chunks corpus | Recall@1 | Recall@3 | Recall@5 | MRR | Indexation | Req. moyenne |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **RAG from-scratch** (v3) | BM25 + métadonnées | 366 | 0.800 | 0.933 | 1.000 | 0.929 | 0.01 s | ~0.5 ms |
| **LlamaIndex** (v7) | Dense vectoriel (`bge-m3`) | 383 | 0.818 | 0.909 | 1.000 | 0.933 | 37.53 s | 45.7 ms |

*Note sur la granularité* : LlamaIndex produit 383 chunks avec `SentenceSplitter(512)` sur les fichiers bruts, contre 366 chunks pour le découpage maison par sections de 400 mots.

## 2. Analyse de la Génération et de l'Abstention

Fiche d'évaluation détaillée : `data/evaluations/llamaindex_review_dev.jsonl`.

### Comportement sur les questions d'abstention du jeu dev

| Question ID | Sujet du piège | Réponse LlamaIndex | Abstention réussie ? |
| --- | --- | --- | --- |
| `q15` | Quelle est la date exacte du prochain examen IA04 ... | Je ne peux pas le déterminer à partir du corpus IA04 fourni. | **Oui (Abstention)** |
| `q16` | Combien d’étudiants étaient inscrits en IA04 à l’automne 2025 ? | Je ne peux pas le déterminer à partir du corpus IA04 fourni. | **Oui (Abstention)** |
| `q24` | Quel problème d'appariement a été posé lors de l'examen médian A26 ? | Je ne peux pas le déterminer à partir du corpus IA04 fourni. | **Oui (Abstention)** |
| `q26` | Comment est implémenté l'algorithme de consensus Raft en Go ? | Je ne peux pas le déterminer à partir du corpus IA04 fourni. | **Oui (Abstention)** |
| `q29` | Sur quel numéro de port TCP le serveur de pong écoute-t-il par défaut ? | Je ne peux pas le déterminer à partir du corpus IA04 fourni... (analyse l'absence de port dans l'énoncé) | **Oui (Abstention)** |

## 3. Comparaison d'Ingénierie & Complexité

| Dimension | RAG from-scratch | LlamaIndex |
| --- | --- | --- |
| **Lignes de code (cœur)** | ~990 lignes | ~30 lignes |
| **Dépendances tierces** | 0 framework RAG (stdlib Python + `pdftotext`) | ~80 paquets installés (`llama-index-core`, `pydantic`, `sqlalchemy`, etc.) |
| **Vitesse d'indexation** | Instantanée (< 0.02s en RAM pour BM25) | 37.5s (calcul de 383 embeddings BGE-M3 sur GPU/CPU) |
| **Découpage documentaire** | Découpage fenêtres 400 mots + 50 overlap (366 chunks) | `SentenceSplitter(512, 50)` LlamaIndex (383 chunks) |
| **Transparence & Débogage** | Totale : chaque formule (IDF, BM25, prompt) est explicite | Boîte noire : abstractions imbriquées (`Node`, `Synthesizer`, etc.) |

## 4. Conclusion

1. **Qualité de la recherche (Retrieval)** : 
   - LlamaIndex avec embeddings denses `bge-m3` obtient un **Recall@5 de 1.000**, un **Recall@1 de 0.818** et un **MRR de 0.933** sur les questions répondables du split `dev`.
   - Ces résultats sont quasiment identiques à notre BM25 avec métadonnées (Recall@5 = 1.000, Recall@1 = 0.800, MRR = 0.929).
2. **Abstention & Robustesse aux distracteurs (Génération)** :
   - C'est le point d'enseignement majeur : **LlamaIndex réussit 5 abstentions sur 5 (100 % de réussite)** sur le split `dev`.
   - En particulier sur **`q16`**, qui provoquait une hallucination tenace dans notre baseline initiale (le modèle affirmait qu'il y avait 24 étudiants en reprenant un exercice de TD sur une machine à café), la recherche dense BGE-M3 combinée au `PromptTemplate` strict de LlamaIndex permet au modèle de s'abstenir sans inventer de chiffre.
   - Sur **`q29`** (port TCP par défaut inexistant), le modèle explique même avec pertinence que le port 8888 apparaît dans un exemple de code mais n'est pas le port officiel du sujet.
3. **Compromis d'ingénierie** :
   - LlamaIndex réduit le volume de code par ~35×, mais ajoute un écosystème lourd (~80 paquets) et une dépendance forte au calcul GPU pour l'indexation (37s vs temps réel en BM25).
