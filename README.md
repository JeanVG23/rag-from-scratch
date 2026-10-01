# RAG construit à la main — IA04

Ce projet explore, étape par étape, la construction en Python d’un système de génération augmentée par récupération (RAG). Le but est de comprendre et d’implémenter les principales briques du système, puis de mesurer ce qu’elles apportent. Le résultat doit rester lisible, reproductible et présentable comme projet personnel.

Le premier corpus étudié est **IA04 — Systèmes multi-agents**. Le projet commencera par un outil en ligne de commande. Les documents des autres cours présents dans `data/raw/` ne feront pas partie de l’index IA04.

## Objectifs

- Construire le pipeline RAG en comprenant le rôle de chaque brique : ingestion, découpage, indexation, recherche, assemblage du contexte et génération.
- Commencer par une recherche simple, puis améliorer le système par petites étapes mesurées.
- Citer les documents utilisés et leurs pages ou sections quand cette information est disponible.
- Évaluer séparément la qualité de la recherche et celle des réponses générées.
- Conserver le code, les paramètres, les résultats et les limites de chaque expérience.
- Comparer, en fin de parcours, l’implémentation manuelle à une solution qui s’appuie sur des bibliothèques spécialisées.

## Corpus de départ

Le corpus IA04 actuellement présent contient **20 documents** :

- 3 supports de cours en Markdown ;
- 11 sujets de TD en Markdown ;
- 6 annales d’examen en PDF.

## Organisation des données

Le traitement suivra une organisation inspirée de l’architecture médaillon, en gardant les étapes simples et inspectables :

- `data/raw/` contient les documents originaux, conservés tels quels ;
- `data/silver/` contiendra leur texte extrait et normalisé, avec les métadonnées de provenance ;
- `data/gold/` contiendra les passages découpés et leurs métadonnées, prêts pour la recherche ;
- `data/indexes/` contiendra les index propres aux méthodes de recherche testées.

Ces données et leurs dérivés restent locaux et sont ignorés par Git. Les index sont des artefacts reconstruisibles ; les paramètres et résultats utiles aux comparaisons seront consignés dans `experiments/`.

## Utilisation locale

Depuis la racine du dépôt, construire les chunks IA04 puis lancer une recherche ou une génération augmentée :

```sh
# 1. Construction des unités et découpage des chunks
PYTHONPATH=src python3 -m rag_from_scratch.cli build-chunks

# 2. Recherche (Hybride RRF par défaut, ou --retriever bm25 / dense)
PYTHONPATH=src python3 -m rag_from_scratch.cli search "Comment les agents communiquent-ils ?" --top-k 5
PYTHONPATH=src python3 -m rag_from_scratch.cli search "Quel problème d'appariement est posé au médian A21 ?" --retriever hybrid

# 3. Génération augmentée avec citations et consigne stricte d'abstention
OLLAMA_CHAT_MODEL=qwen3.5:4b PYTHONPATH=src python3 -m rag_from_scratch.cli ask "Quel problème d'appariement est posé au médian A21 ?"

# 4. Évaluations quantitatives
# Comparaison de recherche (BM25 vs Dense vs Hybride)
PYTHONPATH=src python3 evaluation/run_hybrid.py --questions evaluation/private/ia04_questions_test.jsonl

# Évaluation de bout en bout de la génération et de l'abstention
OLLAMA_CHAT_MODEL=qwen3.5:4b PYTHONPATH=src python3 evaluation/run_generation.py --questions evaluation/private/ia04_questions_test.jsonl --retriever hybrid --overwrite

# Baseline externe de comparaison (LlamaIndex)
PYTHONPATH=src python3 evaluation/run_llamaindex.py
```

### Options du CLI

- `search` et `ask` acceptent `--retriever {hybrid,bm25,dense}` (activé sur `hybrid` par défaut).
- `--top-k` spécifie le nombre de passages renvoyés ou fournis au contexte (par défaut `5`).
- `--dense-model` spécifie le modèle d'embeddings Ollama (par défaut `bge-m3`).
- `--rrf-k` ajuste la constante de lissage de la fusion par rang réciproque (par défaut `60`).
- `--min-bm25-score` permet d'activer un seuil minimal de score en mode BM25.
- La génération (`ask`) fonctionne par défaut à température `0`, avec thinking désactivé et une limite de 384 tokens pour une exécution déterministe et rapide.

## Approche d’implémentation

Dans les premières étapes, les briques principales seront écrites à la main, sans framework RAG, base vectorielle ni bibliothèque de recherche qui masque leur fonctionnement. Les bibliothèques de la collection standard de Python pourront être utilisées quand elles ne remplacent pas une brique étudiée.

Les premières versions sont des **baselines de recherche**. La commande `ask` ajoute maintenant une première génération locale avec les passages récupérés comme contexte et des références dans la réponse ; sa qualité doit encore être mesurée avec la grille du protocole.

## Itérations prévues

Chaque étape devra rester exécutable et avoir une évaluation associée.

1. **Corpus et questions d’évaluation** — inventorier les sources, préparer des questions représentatives et noter les documents ou passages attendus.
2. **Baseline naïve** — découper simplement les documents et classer les passages à partir des mots communs avec la question. Cette version servira de référence et restera archivée.
3. **Ingestion et découpage** — mieux préserver les titres, pages, métadonnées, frontières de sections et chevauchements entre passages.
4. **Recherche lexicale** — implémenter et comparer des méthodes comme TF-IDF et BM25, sans déléguer le classement à une bibliothèque spécialisée.
5. **Assemblage et génération** — première version disponible avec BM25 et Ollama local ; évaluer couverture, fidélité au contexte, citations et abstention avec la grille dédiée.
6. **Comparaison** — mesurer les variantes manuelles sur le même corpus et les mêmes questions, puis les comparer à une solution utilisant des bibliothèques établies.

Cet ordre pourra évoluer si les évaluations montrent qu’une autre amélioration est prioritaire. Les changements de méthode et leurs raisons seront consignés.

## Évaluation

Un jeu de questions préparé à l’avance permettra de comparer les versions sur des bases identiques. Pour la recherche, on pourra suivre notamment le rappel des sources attendues dans les `k` premiers résultats (Recall@k) et le rang réciproque moyen du premier résultat pertinent (MRR). Pour les réponses générées, on vérifiera si elles répondent à la question, s’appuient sur les passages fournis et citent correctement leurs sources. L’évaluation du système reste un objectif ; la publication d’un jeu de test indépendant des documents de cours dépendra du temps disponible.

Les résultats devront inclure les paramètres utilisés et les erreurs observées, pas seulement quelques exemples réussis. Les questions et critères de référence seront gardés fixes pendant la comparaison des itérations ; toute évolution du jeu d’évaluation sera elle aussi documentée.

## Historique et archivage des expériences

Git conservera l’historique du code. Chaque version importante pourra être identifiée par un tag, par exemple `rag-v0-naive` ou `rag-v1-bm25`. Un rapport associé consignera :

- l’objectif et les changements de l’itération ;
- le commit ou tag du code utilisé ;
- les paramètres et la version du corpus ;
- les résultats d’évaluation ;
- les limites constatées et la motivation de l’étape suivante.

Cette méthode garde les premières implémentations consultables et comparables sans maintenir plusieurs copies ambiguës du même code.

## Périmètre et limites

- Le développement initial porte uniquement sur IA04, même si d’autres cours sont déjà présents dans `data/raw/`.
- Le premier moteur lexical retrouvera surtout les passages qui partagent des mots avec la question. Il ne saisira pas nécessairement les paraphrases ou les synonymes.
- La qualité de l’extraction PDF et des figures ou tableaux sera examinée séparément ; une conversion en texte peut perdre de la structure.
- L’entraînement d’un grand modèle de langage ou d’un modèle d’embeddings n’est pas un objectif. L’utilisation d’un modèle sera ajoutée comme composant distinct, afin de garder visible le code du pipeline RAG.

## Données et diffusion

Les documents de cours et les données dérivées ne seront pas distribués avec le projet. Le dossier `data/` est ignoré par Git via `.gitignore` ; `data/raw/` contiendra les sources utilisées pendant le développement. Le code public devra expliquer où placer les documents localement et comment lancer l’indexation, sans inclure leur contenu.

## État du projet (Version finale v9)

Le projet a atteint l'ensemble de ses objectifs initiaux et dispose d'une suite complète d'expérimentations reproductibles :

1. **Ingestion & Découpage hiérarchique** : 20 documents IA04 (334 sections Markdown, 23 pages PDF) segmentés en 366 chunks préservant les chemins de sections et numéros de pages.
2. **Recherche Hybride From-Scratch** : Intégration de BM25 avec métadonnées et d'une recherche vectorielle dense (`bge-m3`), fusionnées par *Reciprocal Rank Fusion* (RRF, $k=60$).
3. **Génération & Consigne stricte d'abstention** : Modèle local Ollama (`qwen3.5:4b`) avec citations déterministes `[S1]`, `[S2]` et consigne stricte éliminant les hallucinations sur les distracteurs et questions sans réponse.
4. **Protocole d'évaluation scellé** : Jeu de 30 questions partitionné en sous-ensembles étanches `dev` (calibrage) et `test` (validation finale en aveugle).

### Tableau comparatif des performances (Split `test` indépendant)

| Système | Méthode d'indexation | Recall@3 | Recall@5 | MRR | Abstention (Distracteurs) | Dépendances tierces |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **BM25 baseline (v3)** | Lexical TF-IDF RAM | 1.000 | 1.000 | 1.000 | 0 % *(hallucinait les distracteurs)* | 0 framework |
| **LlamaIndex baseline (v7)** | Dense vectoriel (`bge-m3`) | 0.909 | 1.000 | 0.933 | 100 % (5/5) | ~80 paquets Python |
| **Notre RAG Hybride (v9)** | **BM25 + Dense RRF ($k=60$)** | **1.000** | **1.000** | **0.950** | **100 % (5/5)** | **0 framework** |

Les rapports détaillés de chaque itération sont consultables dans le répertoire `experiments/` (de `v1` à `v9`).
