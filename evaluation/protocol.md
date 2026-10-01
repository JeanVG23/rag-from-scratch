# Protocole d’évaluation

## Périmètre et jeux d’évaluation

L’évaluation porte sur le corpus IA04. Le jeu de questions compte **30 questions** au total, partitionnées en deux jeux équilibrés de 15 questions chacun (`dev` pour la calibration et le développement, `test` pour l’évaluation finale) :

- **20 questions répondables** (`q01`–`q14`, `q17`–`q22`) couvrant les concepts multi-agents, la programmation concurrente en Go, la théorie du choix social (vote), la théorie des jeux, les enchères et les annales d’examen.
- **10 questions sans réponse** (`q15`–`q16`, `q23`–`q30`) conçues pour évaluer l’abstention face à des distracteurs variés :
  - distracteurs numériques (effectif d’étudiants, tarifs de machine à café, moyennes d’examen, ports réseau) ;
  - distracteurs temporels (examens de sessions futures, dates non mentionnées) ;
  - distracteurs sémantiques ou inter-cours (algorithmes absents comme Raft, organisation d’autres UV comme LO23, emprunt mémoire en Rust).

Les questions complètes et leurs références sont conservées localement dans `evaluation/private/ia04_questions.jsonl` (avec le champ `"split"` valant `"dev"` ou `"test"`), ainsi que dans les deux fichiers partitionnés :
- `evaluation/private/ia04_questions_dev.jsonl` : 15 questions (10 répondables, 5 d’abstention) ;
- `evaluation/private/ia04_questions_test.jsonl` : 15 questions (10 répondables, 5 d’abstention).

Ces fichiers restent ignorés par Git avec le contenu des cours. Les hyperparamètres (seuils de score, poids des champs BM25, prompts de vérification) doivent être calibrés sur le jeu `dev` avant toute mesure sur le jeu `test`.

Pour chaque question répondable, la référence indique un document et une page ou une section. On évalue donc la capacité du moteur à retrouver une zone pertinente, sans dépendre d’un découpage en passages qui va évoluer. Les questions sans réponse sont exclues de Recall@k et du MRR ; elles servent à mesurer la capacité du système à s’abstenir sans halluciner.

## Mesures de recherche

- **Recall@k** : proportion des références pertinentes retrouvées dans les `k` premiers résultats.
- **MRR** (*Mean Reciprocal Rank*) : moyenne de l’inverse du rang du premier résultat pertinent.

Le rapport d’une expérience précisera les valeurs de `k`, les paramètres de recherche, le temps d’exécution, le nombre de questions évaluées et les erreurs représentatives. Les comparaisons garderont le même corpus, le même jeu de questions et les mêmes règles de pertinence.

## Génération

Les références actuelles évaluent le rappel des sources, pas l’exactitude des réponses générées. Le jeu privé est enrichi avec `expected_answer_points`, une liste de faits atomiques attendus. Chaque point porte les références document/page/section qui le justifient. Les questions sans réponse ont une liste vide, `expected_abstention: true` et une `abstention_reason`.

Les questions et réponses de référence restent privées. Pour les questions d’abstention, le corpus local ne fournit pas les données demandées ; le modèle doit le signaler au lieu d’inventer une date, un nombre ou un algorithme.

### Grille de notation des réponses

Les dimensions sont notées séparément ; on ne les réduit pas à un score global qui masquerait leurs différences.

1. **Couverture des faits attendus** : noter chaque point de `expected_answer_points` : `1` s’il est correctement couvert avec ses précisions essentielles, `0.5` s’il est partiellement correct ou incomplet, `0` s’il manque ou est faux. La couverture d’une question est la moyenne de ses points.
2. **Fidélité au contexte** : `2` si les affirmations factuelles sont appuyées par les passages récupérés, `1` si une affirmation mineure est insuffisamment étayée, `0` si une affirmation centrale est inventée ou contredite par les sources.
3. **Qualité des citations** : `2` si les affirmations importantes sont reliées à une source et à une page/section qui les justifient, `1` si les sources sont globalement pertinentes mais les repères sont incomplets, `0` si les citations manquent ou pointent vers des sources inadéquates.
4. **Abstention** : pour les questions sans réponse, `1` si la réponse dit clairement que le corpus ne permet pas de conclure et n’invente pas de fait, `0` sinon. Pour une question répondable dont les passages récupérés omettent un fait attendu, noter séparément l’abstention prudente et le défaut de couverture ; ne pas récompenser une affirmation non étayée.

Les variantes de formulation sont acceptées si elles expriment le même fait. Une réponse ne doit pas être pénalisée parce qu’elle ne reprend pas les mots exacts du corrigé. Le rapport de génération devra également consigner les erreurs représentatives sans reproduire le contenu privé des cours.

`evaluation/run_generation.py` prépare une fiche locale `data/evaluations/generation_review.jsonl`. Pour chaque question, elle conserve la réponse du modèle, le contexte réellement fourni, les critères attendus et des champs à remplir : `coverage_score` pour chaque fait atomique, puis `groundedness_0_to_2`, `citation_quality_0_to_2` et `abstention_0_or_1`. La fiche contient le texte de passages du cours ; elle reste donc dans `data/`, ignoré par Git. Après génération, noter les réponses selon la grille ci-dessus avant de tirer des conclusions sur la qualité du modèle.

Les paramètres de génération de la première baseline sont une température de `0`, le raisonnement désactivé et une limite de `384` tokens (`--think` et `--num-predict` permettent de les modifier). Ils sont enregistrés dans chaque ligne de la fiche afin que les réponses restent comparables et reproductibles.

## Partitionnement et prévention du sur-apprentissage

Le partitionnement en deux sous-ensembles étanches répond aux limites relevées dans le rapport `experiments/v6-abstention-threshold.md` :
1. **Jeu de développement / calibration (`dev`)** : sert à tester les variations d’ingestion, les poids BM25, les invites du modèle et à calibrer les seuils éventuels d’abstention.
2. **Jeu de test final (`test`)** : reste figé et n’est évalué qu’une fois la configuration arrêtée pour mesurer la performance réelle sans sur-ajustement.

Les documents IA04 ne sont pas diffusés avec le projet. Les rapports publics se limitent aux métriques et aux observations sans reproduire le contenu des cours.
