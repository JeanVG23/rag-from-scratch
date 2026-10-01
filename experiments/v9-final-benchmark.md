# Comparaison de Recherche : BM25 vs Dense vs Hybride (RRF)

Généré le 2026-10-01 08:35 UTC sur `ia04_questions_test.jsonl` (15 questions : 10 répondables, 5 sans réponse).
- Corpus : 366 chunks (`ia04_chunks.jsonl`).
- Modèle Dense : `bge-m3` (embeddings pré-normalisés).
- Constante RRF : $k = 60$.

## 1. Métriques de Recherche (Retrieval)

| Système | Index / Méthode | Recall@1 | Recall@3 | Recall@5 | MRR | Latence moyenne | Initialisation |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BM25** (metadata) | Lexical (TF-IDF RAM) | 1.000 | 1.000 | 1.000 | 1.000 | 0.7 ms | 11.4 ms |
| **Dense** (`bge-m3`) | Vecteurs 1024-d | 0.800 | 0.800 | 0.900 | 0.833 | 186.7 ms | 35.5 ms |
| **Hybride RRF** | Fusion des rangs ($k=60$) | **0.900** | **1.000** | **1.000** | **0.950** | 28.1 ms | 0.0 ms |

## 2. Analyse détaillée par question répondable

| Question ID | Question | BM25 R@1 (Rang) | Dense R@1 (Rang) | Hybride R@1 (Rang) |
| :--- | :--- | :---: | :---: | :---: |
| `q02` | Quels avantages et quels risques sont associés aux systèmes répar... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q04` | Que signifie le fait qu’un channel Go soit asynchrone, bidirectio... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q07` | Comment la règle de Kemeny compare-t-elle un classement candidat ... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q09` | Dans le TD8, quel jeu sert à étudier les équilibres de Nash en st... | ✓ (rang 1) | ✗ (rang 4) | ✓ (rang 1) |
| `q11` | Quel algorithme le TD10 demande-t-il d’appliquer pour échanger de... | ✓ (rang 1) | ✗ (rang 13) | ✗ (rang 2) |
| `q14` | Quel algorithme d’affectation le sujet A24 demande-t-il d’utilise... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q18` | Quelles sont les méthodes requises par l'interface Locker en Go e... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q20` | Dans le TD2bis, quelles sont les durées typiques de la préparatio... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q21` | Dans l'exercice de fusion de croyances du TD7 (examen A21), combi... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q22` | D'après l'examen final A23 d'IA04 (exercice Pokémon), quelle réfé... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |

## 3. Évaluation de la Génération et de l'Abstention (End-to-End sur le split test)

Fiche d'évaluation détaillée : `data/evaluations/generation_review_scratch_test.jsonl`.
Modèle : `qwen3.5:4b` (température 0, thinking désactivé, prompt strict).
Moteur de recherche : **Hybride RRF** ($k=60$).

### A. Comportement sur les 5 questions d'abstention inédites

| Question ID | Sujet du piège (Distracteur) | Réponse générée par le RAG Hybride | Statut |
| :--- | :--- | :--- | :---: |
| `q23` | Prix en centimes d'un café à la machine BF | `Je ne peux pas le déterminer à partir du corpus IA04 fourni.` | **Succès (Abstention)** |
| `q25` | Heures de TD hebdomadaires en LO23 | `Je ne peux pas le déterminer à partir du corpus IA04 fourni.` | **Succès (Abstention)** |
| `q27` | Note moyenne à l'examen final A23 | `Je ne peux pas le déterminer à partir du corpus IA04 fourni.` | **Succès (Abstention)** |
| `q28` | Date précise de la simulation en janvier 2026 | `Je ne peux pas le déterminer à partir du corpus IA04 fourni.` | **Succès (Abstention)** |
| `q30` | Règles de borrowing Rust dans le cours Go | `Je ne peux pas le déterminer à partir du corpus IA04 fourni.` | **Succès (Abstention)** |

**Taux d'abstention sur le split `test` : 5 / 5 (petit échantillon)**.  
Aucune hallucination observée sur les 5 distracteurs inédits.

### B. Comportement sur les 10 questions répondables inédites

* **9 / 10 réponses étayées et citées** avec une conformité totale aux critères atomiques attendus (`q02`, `q04`, `q07`, `q11`, `q14`, `q18`, `q20`, `q21`, `q22`).
* Les citations `[S1]`, `[S2]` pointent systématiquement vers les sections exactes des supports et annales.
* **1 abstention à tort (`q09`)** : question répondable, le modèle a préféré s'abstenir face à une ambiguïté textuelle (le TD8 contient deux jeux étudiant les équilibres purs et mixtes : la guerre des sexes et pierre-feuille-ciseaux).

## 4. Synthèse globale du projet RAG From-Scratch

| Étape de l'architecture | Version | Recall@3 | Recall@5 | MRR | Abstention dev | Abstention test | Dépendances |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **BM25 baseline initiale** | v3 | 0.909 | 1.000 | 0.833 | 0 / 5 (hallucinations) | - | aucun framework RAG |
| **Baseline LlamaIndex** | v7 | 0.909 | 1.000 | 0.933 | 5 / 5 | - | ~80 paquets (LlamaIndex) |
| **RAG From-Scratch Hybride RRF** | **v9** | **1.000** | **1.000** | **0.950** | **5 / 5** | **5 / 5** | **aucun framework RAG** |

*Attention : les métriques de retrieval des lignes v3 et v7 sont mesurées sur `dev`, celles de la ligne v9 sur `test`. Elles ne sont pas directement comparables.*

### Conclusions et limites

1. **Retrieval** : sur `test`, BM25 seul (MRR 1.000, Recall@1 1.000) fait au moins aussi bien que l'hybride RRF (MRR 0.950, Recall@1 0.900). L'avantage de l'hybride n'apparaît que sur `dev` (v8, MRR 0.950 contre 0.833), où il a été choisi. Avec 10 questions répondables par split, la supériorité de l'hybride n'est pas démontrée.
2. **Abstention** : 5/5 sur `test`, mais sur 5 questions seulement, avec une notation manuelle par une seule personne et une consigne de prompt retouchée après les échecs observés sur `dev` (q16, voir v5). Une abstention à tort est aussi observée sur une question répondable (`q09`).
3. **LlamaIndex** n'a pas été évalué sur `test` (v7 porte sur `dev`, avec un découpage de 383 chunks). Les colonnes de la synthèse ci-dessus ne sont donc pas comparables entre elles.
4. **Coût** : la comparaison BM25 (< 0.1 s) contre LlamaIndex (37 s, v7) n'est pas équitable : la seconde inclut le calcul des embeddings, pas la première. Le temps d'indexation de l'hybride, embeddings compris, n'est pas mesuré ici.
5. **Dépendances** : sans framework RAG, mais avec `numpy`, `pdftotext` et Ollama.
