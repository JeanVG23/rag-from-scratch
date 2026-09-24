# Protocole d’évaluation

## Périmètre de départ

La première évaluation porte uniquement sur le corpus IA04. Le jeu pilote local contient 14 questions auxquelles le corpus devrait permettre de retrouver une source, et 2 questions sans réponse dans le corpus. Les questions et leurs références sont conservées dans `evaluation/private/ia04_questions.jsonl`, ignoré par Git avec le contenu des cours.

Pour chaque question répondable, la référence indique un document et une page ou une section. On évalue donc la capacité du moteur à retrouver une zone pertinente, sans dépendre d’un découpage en passages qui va évoluer. Les questions sans réponse sont exclues de Recall@k et du MRR ; elles serviront plus tard à mesurer la capacité du système à s’abstenir.

## Mesures de recherche

- **Recall@k** : proportion des références pertinentes retrouvées dans les `k` premiers résultats.
- **MRR** (*Mean Reciprocal Rank*) : moyenne de l’inverse du rang du premier résultat pertinent.

Le rapport d’une expérience précisera les valeurs de `k`, les paramètres de recherche, le temps d’exécution, le nombre de questions évaluées et les erreurs représentatives. Les comparaisons garderont le même corpus, le même jeu de questions et les mêmes règles de pertinence.

## Génération

Ces références évaluent le rappel de sources, pas l’exactitude des réponses générées. Avant d’évaluer un générateur, il faudra ajouter au jeu privé des éléments de réponse attendus et une grille vérifiant la fidélité aux passages récupérés et la qualité des citations.

## Limites du jeu pilote

Le jeu est petit et conçu à la main ; il sert à comparer les premières itérations, pas à établir une mesure générale. Les documents IA04 ne sont pas diffusés avec le projet. Les rapports publics devront se limiter aux métriques et aux observations qui ne reproduisent pas leur contenu.
