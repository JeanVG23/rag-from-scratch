# Génération IA04 — première revue

Revue produite le 26 septembre 2026 à partir de la fiche locale de génération.

## Configuration

- Modèle local Ollama : `qwen3.5:4b`.
- Recherche : BM25, `top_k=5`, `k1=1.5`, `b=0.75`, texte et métadonnées indexés.
- Génération : température `0`, raisonnement désactivé, `num_predict=384`.
- Questions : 16 au total, dont 14 répondables et 2 conçues pour mesurer l’abstention.
- Faits atomiques de référence : 28.
- Les réponses, questions, passages et annotations détaillées restent dans `data/evaluations/generation_review.jsonl`, fichier local ignoré par Git.

## Résultats

| Dimension | Résultat | Calcul |
| --- | ---: | --- |
| Couverture des faits | 0,798 | Moyenne des couvertures par question répondable, de 0 à 1 |
| Fidélité au contexte | 1,438 / 2 | Moyenne sur les 16 réponses |
| Qualité des citations | 1,625 / 2 | Moyenne sur les 16 réponses |
| Abstention | 1 / 2 | Questions sans réponse dans le corpus |

Quinze réponses sur seize contiennent au moins un repère de citation ; aucun repère ne pointe vers un identifiant absent du contexte. Cette vérification mécanique ne remplace pas la notation de la pertinence des références.

## Erreurs relevées

- `q01` ne restitue pas la distinction entre un agent et un système multi-agent ; la réponse se concentre sur le nombre minimal d’agents et les propriétés du système.
- `q03` décrit `n++` de manière contradictoire sur son atomicité.
- `q04` contredit l’exemple cité sur le blocage d’un envoi dans un channel bufferisé.
- `q08` transforme une condition suffisante en condition nécessaire et suffisante.
- `q13` confond les deux algorithmes demandés et cite aussi un sujet différent.
- `q16` affirme un effectif à partir d’un passage d’exercice sans rapport avec les inscriptions, au lieu de s’abstenir.

## Limites

- Il s’agit d’un jeu pilote de 16 questions ; les scores sont indicatifs et ne permettent pas de généraliser à d’autres cours ou corpus.
- Les notes détaillées ont été attribuées par revue des réponses, des critères et des passages récupérés. Elles comportent donc une part de jugement.
- Un résultat BM25 peut contenir un passage lexicalement proche mais sans rapport avec la question, comme le montre `q16`.
- La limite de sortie de 384 tokens et le raisonnement désactivé font partie de cette baseline ; modifier ces paramètres produit une autre configuration à évaluer.
- L’évaluation mesure la réponse après récupération BM25. Elle ne sépare pas entièrement une erreur de génération d’un fait absent des passages récupérés.
