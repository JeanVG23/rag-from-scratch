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

Depuis la racine du dépôt, construire les chunks IA04 puis lancer une recherche BM25 :

```sh
PYTHONPATH=src python3 -m rag_from_scratch.cli build-chunks
PYTHONPATH=src python3 -m rag_from_scratch.cli search "Comment les agents communiquent-ils ?" --top-k 5
OLLAMA_CHAT_MODEL=qwen3.5:4b PYTHONPATH=src python3 -m rag_from_scratch.cli ask "Comment les agents communiquent-ils ?"
PYTHONPATH=src python3 evaluation/run_bm25.py
PYTHONPATH=src python3 evaluation/analyze_bm25.py
PYTHONPATH=src python3 evaluation/run_bm25.py --include-metadata
OLLAMA_CHAT_MODEL=qwen3.5:4b PYTHONPATH=src python3 evaluation/run_generation.py
```

La commande `search` lit `data/gold/ia04_chunks.jsonl` par défaut et indexe le texte, le nom du fichier source et le titre de section. `--content-only` rétablit la recherche de la baseline. On peut choisir un autre fichier avec `--chunks`, ainsi que modifier `--top-k`, `--k1` et `--b`. Les résultats affichent le score BM25, la source et la page ou section quand elle est disponible. `evaluation/analyze_bm25.py` produit l’analyse des variantes dans `experiments/v2-bm25-analysis.md`. Le script d’évaluation réutilise le jeu privé de questions : sans option, il écrit la baseline dans `experiments/v2-bm25.md` ; avec `--include-metadata`, il écrit `experiments/v3-bm25-metadata.md`.

La commande `ask` assemble les cinq premiers passages BM25 avec leurs repères `[S1]`, `[S2]`, etc., puis appelle l’API locale `/api/chat` d’Ollama. Le modèle est choisi avec `OLLAMA_CHAT_MODEL` ou `--model` ; l’hôte peut être changé avec `OLLAMA_HOST` ou `--host`. La génération demande une réponse en français, fondée sur les seuls passages récupérés, et une abstention lorsque le contexte ne permet pas de répondre. Par défaut, la température est `0`, le raisonnement est désactivé et la sortie est limitée à 384 tokens ; `--think` l’active et `--num-predict` change la limite. Les références fournies au modèle sont affichées après sa réponse.

`evaluation/run_generation.py` génère une réponse pour chaque question, y compris celles conçues pour mesurer l’abstention. Il produit `data/evaluations/generation_review.jsonl`, une fiche privée contenant la réponse, les passages avec leur texte, les critères de référence et des champs vides pour la notation. Ce fichier peut contenir du contenu des supports et reste dans `data/`, ignoré par Git. Les paramètres de recherche et de génération sont inscrits dans chaque ligne. Les dimensions à noter et leurs échelles sont décrites dans `evaluation/protocol.md`. Une fiche existante n’est pas remplacée par défaut ; utiliser `--overwrite` la recrée et efface les scores et notes qui s’y trouveraient. `--limit 1` permet de générer uniquement la première question pour vérifier la configuration du modèle. La première revue agrégée, sans question ni extrait du cours, se trouve dans `experiments/v4-generation.md`.

Les PDF IA04 contiennent du texte extractible. La première ingestion utilise `pdftotext -layout` page par page et conserve le numéro de page. L’audit montre que l’ordre de lecture n’est pas encore validé partout : certaines pages ont plusieurs colonnes, des tableaux ou des matrices. Une itération ultérieure pourra exploiter les coordonnées de `pdftotext -bbox-layout` pour rétablir l’ordre, en gardant les tableaux et matrices comme des blocs. Le contenu uniquement graphique qui n’apparaît pas dans la couche texte du PDF ne sera pas indexé.

Les documents LO23 sont hors du premier périmètre. Certains PDF LO23 présentent plusieurs diapositives sur une même page. Si on les intègre plus tard, l’extraction utilisera les coordonnées fournies par `pdftotext` pour regrouper le texte par diapositive et garder la page PDF d’origine comme référence. Le contenu absent de la couche texte ne sera pas indexé.

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

## État du projet

Le corpus IA04 a été inventorié et les PDF ont fait l’objet d’un contrôle d’extraction ; le rapport est dans `evaluation/corpus_audit.md`. L’ingestion actuelle produit 357 unités de texte (334 sections Markdown et 23 pages PDF), puis 366 chunks de 400 mots avec 50 mots de chevauchement. Ces artefacts restent dans `data/silver/` et `data/gold/`, ignorés par Git. Le jeu pilote privé contient 14 questions répondables et 2 sans réponse. Le rapport `experiments/v1-embedding.md` compare la recherche par similarité cosinus avec BGE-M3, Qwen3-Embedding 0.6B et Jina Embeddings v5 sur les mêmes chunks et les mêmes questions. L’ordre de lecture des pages PDF n’est pas encore validé sur tout le corpus.
