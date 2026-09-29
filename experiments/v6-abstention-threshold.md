# Analyse exploratoire d’un seuil BM25 pour l’abstention

Seuil fixe testé : `12`. Jeu pilote privé : 30 questions.
Le seuil prédit « répondable » si le score BM25 top 1 est supérieur ou égal à cette valeur.
Les résultats ne sont pas une mesure indépendante : le seuil postérieur indiqué est calculé sur le même jeu.

| Variante | Score min. répondable | Score max. négatif | Écart | q16 | Seuil fixe TP/FN/FP/TN | Milieu post hoc TP/FN/FP/TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 courant : texte + métadonnées concaténés | 16.267 | 36.627 | -20.360 | 9.875 | 20/0/8/2 | aucune séparation |
| Texte seulement | 14.763 | 36.742 | -21.979 | 11.115 | 20/0/8/2 | aucune séparation |
| Métadonnées et mots fréquents retirés de la requête | 14.574 | 33.705 | -19.131 | 6.953 | 20/0/7/3 | aucune séparation |
| BM25 texte + 0.25 × métadonnées, index séparés | 16.384 | 41.596 | -25.212 | 12.157 | 20/0/9/1 | aucune séparation |
| BM25 texte + 0.50 × métadonnées, index séparés | 18.282 | 46.450 | -28.168 | 13.199 | 20/0/9/1 | aucune séparation |
| BM25 texte + 1.00 × métadonnées, index séparés | 22.139 | 56.157 | -34.018 | 15.284 | 20/0/10/0 | aucune séparation |

Dans la colonne de confusion, l’ordre est `TP/FN/FP/TN`. Un seuil appris sur ces 16 questions serait optimiste : il n’y a que deux questions sans réponse et aucune partition de calibration distincte.
Les scores changent quand le mode BM25 ou le poids des champs change. Le seuil fixe n’est donc valable que pour une configuration figée, et les chiffres de ce jeu ne justifient pas son activation par défaut.

Détails agrégés sans texte privé : `data/evaluations/abstention_threshold.json`.
