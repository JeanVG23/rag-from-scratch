# Protocole d’évaluation

## Périmètre de départ

La première évaluation porte uniquement sur le corpus IA04. Le jeu pilote local contient 14 questions auxquelles le corpus devrait permettre de retrouver une source, et 2 questions sans réponse dans le corpus. Les questions et leurs références sont conservées dans `evaluation/private/ia04_questions.jsonl`, ignoré par Git avec le contenu des cours.

Pour chaque question répondable, la référence indique un document et une page ou une section. On évalue donc la capacité du moteur à retrouver une zone pertinente, sans dépendre d’un découpage en passages qui va évoluer. Les questions sans réponse sont exclues de Recall@k et du MRR ; elles serviront plus tard à mesurer la capacité du système à s’abstenir.

## Mesures de recherche

- **Recall@k** : proportion des références pertinentes retrouvées dans les `k` premiers résultats.
- **MRR** (*Mean Reciprocal Rank*) : moyenne de l’inverse du rang du premier résultat pertinent.

Le rapport d’une expérience précisera les valeurs de `k`, les paramètres de recherche, le temps d’exécution, le nombre de questions évaluées et les erreurs représentatives. Les comparaisons garderont le même corpus, le même jeu de questions et les mêmes règles de pertinence.

## Génération

Les références actuelles évaluent le rappel des sources, pas l’exactitude des réponses générées. Le jeu privé est enrichi avec `expected_answer_points`, une liste de faits atomiques attendus. Chaque point porte les références document/page/section qui le justifient. Les questions sans réponse ont une liste vide, `expected_abstention: true` et une `abstention_reason`.

Les questions et réponses de référence restent privées. Pour q15 et q16, le corpus local ne fournit respectivement ni calendrier du prochain examen ni données d’effectif pour l’automne 2025 ; le modèle doit le signaler au lieu d’inventer une date ou un nombre.

### Grille de notation des réponses

Les dimensions sont notées séparément ; on ne les réduit pas à un score global qui masquerait leurs différences.

1. **Couverture des faits attendus** — noter chaque point de `expected_answer_points` : `1` s’il est correctement couvert avec ses précisions essentielles, `0.5` s’il est partiellement correct ou incomplet, `0` s’il manque ou est faux. La couverture d’une question est la moyenne de ses points.
2. **Fidélité au contexte** — `2` si les affirmations factuelles sont appuyées par les passages récupérés, `1` si une affirmation mineure est insuffisamment étayée, `0` si une affirmation centrale est inventée ou contredite par les sources.
3. **Qualité des citations** — `2` si les affirmations importantes sont reliées à une source et à une page/section qui les justifient, `1` si les sources sont globalement pertinentes mais les repères sont incomplets, `0` si les citations manquent ou pointent vers des sources inadéquates.
4. **Abstention** — pour q15 et q16, `1` si la réponse dit clairement que le corpus ne permet pas de conclure et n’invente pas de fait, `0` sinon. Pour une question répondable dont les passages récupérés omettent un fait attendu, noter séparément l’abstention prudente et le défaut de couverture ; ne pas récompenser une affirmation non étayée.

Les variantes de formulation sont acceptées si elles expriment le même fait. Une réponse ne doit pas être pénalisée parce qu’elle ne reprend pas les mots exacts du corrigé. Le rapport de génération devra également consigner les erreurs représentatives sans reproduire le contenu privé des cours.

## Limites du jeu pilote

Le jeu est petit et conçu à la main ; il sert à comparer les premières itérations, pas à établir une mesure générale. Les documents IA04 ne sont pas diffusés avec le projet. Les rapports publics devront se limiter aux métriques et aux observations qui ne reproduisent pas leur contenu.
