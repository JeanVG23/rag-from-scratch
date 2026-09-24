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

Les PDF IA04 contiennent du texte extractible. L’ingestion utilisera `pdftotext -bbox-layout`, qui fournit les lignes de texte avec leurs coordonnées. Un traitement Python léger rétablira l’ordre des colonnes ou des vignettes à partir de ces coordonnées. Il devra conserver les tableaux et les matrices comme des blocs, car un simple tri par abscisse en mélange les cellules. Les références conserveront la page PDF et, si nécessaire, la zone de la page. Le contenu uniquement graphique qui n’apparaît pas dans la couche texte du PDF ne sera pas indexé.

Les documents LO23 sont hors du premier périmètre. Certains PDF LO23 présentent plusieurs diapositives sur une même page. Si on les intègre plus tard, l’extraction utilisera les coordonnées fournies par `pdftotext` pour regrouper le texte par diapositive et garder la page PDF d’origine comme référence. Le contenu absent de la couche texte ne sera pas indexé.

## Approche d’implémentation

Dans les premières étapes, les briques principales seront écrites à la main, sans framework RAG, base vectorielle ni bibliothèque de recherche qui masque leur fonctionnement. Les bibliothèques de la collection standard de Python pourront être utilisées quand elles ne remplacent pas une brique étudiée.

Les premières versions seront des **baselines de recherche** : elles retrouveront et afficheront des passages pertinents, sans prétendre encore produire des réponses générées. Une version ultérieure ajoutera un modèle de génération, avec les passages récupérés comme contexte et des références dans la réponse.

## Itérations prévues

Chaque étape devra rester exécutable et avoir une évaluation associée.

1. **Corpus et questions d’évaluation** — inventorier les sources, préparer des questions représentatives et noter les documents ou passages attendus.
2. **Baseline naïve** — découper simplement les documents et classer les passages à partir des mots communs avec la question. Cette version servira de référence et restera archivée.
3. **Ingestion et découpage** — mieux préserver les titres, pages, métadonnées, frontières de sections et chevauchements entre passages.
4. **Recherche lexicale** — implémenter et comparer des méthodes comme TF-IDF et BM25, sans déléguer le classement à une bibliothèque spécialisée.
5. **Assemblage et génération** — fournir les meilleurs passages à un modèle, demander une réponse fondée sur ces sources et afficher les citations. La réponse devra pouvoir signaler que le corpus ne permet pas de conclure.
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

Le corpus IA04 a été inventorié et les PDF ont fait l’objet d’un contrôle d’extraction ; le rapport est dans `evaluation/corpus_audit.md`. `pdftotext -bbox-layout` extrait du texte des 23 pages IA04, mais l’ordre géométrique n’est pas encore validé sur tout le corpus : les colonnes doivent être distinguées des tableaux et matrices. Le manifeste local `data/manifest.jsonl` enregistre les sources et leurs empreintes, et un jeu pilote privé de 16 questions est prêt pour mesurer la recherche. L’ingestion complète, le découpage et la première baseline restent à construire.
