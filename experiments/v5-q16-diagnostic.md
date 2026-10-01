# Diagnostic de la question q16 : réponse inventée au lieu d’une abstention

Diagnostic initial réalisé le 26 septembre 2026 et complété le 27 septembre à partir de la fiche locale de génération et des 366 chunks du corpus.

## Résumé

La question q16 est étiquetée comme sans réponse dans le corpus : aucune donnée locale ne permet d’établir l’effectif inscrit en IA04 à l’automne 2025. Pourtant, le système répond avec un nombre précis et une citation.

La cause est une chaîne en deux temps : BM25 sélectionne un passage qui partage les termes « IA04 » et « étudiants », mais décrit une situation sans rapport avec des inscriptions ; le modèle reprend le nombre de ce passage et le présente comme l’effectif demandé. La citation existe bien dans le contexte, mais elle ne justifie pas l’affirmation.

## Configuration examinée

- Recherche : BM25 sur le texte et les métadonnées, `top_k=5`, `k1=1.5`, `b=0.75`.
- Génération : `qwen3.5:4b`, température `0`, raisonnement désactivé, limite de `384` tokens.
- Référence d’évaluation : question non répondable, abstention attendue.
- Notation manuelle de q16 : fidélité au contexte `0/2`, qualité de citation `0/2`, abstention `0/1`.

## Ce que la recherche a fourni

| Rang | Score BM25 | Type de document et sujet du passage |
| ---: | ---: | --- |
| 1 | 9,875 | TD sur la synchronisation ; exercice de machine à café mentionnant un groupe d’étudiants |
| 2 | 9,414 | Questions du même exercice de TD |
| 3 | 7,819 | Annale sur les préférences et les appariements |
| 4 | 7,552 | Annale sur le vote et les profils de préférence |
| 5 | 7,226 | TD sur le vote et son arborescence de code |

Le premier résultat n’est donc pas une donnée d’inscription. C’est un exemple de TD, et son nombre décrit uniquement la situation de l’exercice. Les deux premiers résultats proviennent en plus de deux chunks voisins du même document : ils occupent deux des cinq places sans apporter de preuve sur l’effectif demandé.

## Pourquoi BM25 l’a remonté

Après normalisation, les termes de la question liés à l’inscription, à la période et à l’année : `inscrits`, `automne`, `2025` : ne correspondent pas au premier passage. Les correspondances utiles sont surtout `IA04` et `étudiants`. La tokenisation conserve aussi des mots très fréquents comme `a`, `d`, `en` et `l`, sans stemming ni liste de mots vides.

Les métadonnées sont concaténées au texte et reçoivent le même poids. Le titre de section contenant `IA04` contribue donc au score, même s’il ne dit rien sur les inscriptions. BM25 classe ici une proximité lexicale ; son score ne mesure ni la présence de la relation « être inscrit », ni la correspondance de la période « automne 2025 ». Le score de `9,875` ne peut pas servir de preuve qu’une réponse existe.

## Pourquoi la génération a échoué

Le prompt demandait de répondre uniquement à partir des passages et de s’abstenir si le contexte ne justifie pas le fait demandé. Le modèle n’a pas respecté cette consigne : il a pris le nombre du scénario de TD, l’a réinterprété comme un effectif officiel et l’a associé à l’automne 2025, période absente du passage.

Le repère `[S1]` est syntaxiquement valide puisqu’il existe dans le contexte, mais la source n’étaye ni l’inscription ni la période. La validation mécanique des identifiants de citation ne suffit donc pas à détecter une citation hors sujet. Le défaut principal côté génération est une absence de vérification de l’entaillement : le passage contient un nombre et parle d’étudiants, mais il ne répond pas à la relation précise demandée.

## Diagnostic par composant

| Composant | Constat | Rôle dans l’échec |
| --- | --- | --- |
| Corpus | Pas de donnée d’inscription pour cette période | Une réponse factuelle ne devrait pas être produite |
| BM25 | Passage lexicalement proche mais hors sujet ; chunks voisins redondants | Introduit un nombre distracteur et donne un contexte trompeur |
| Génération | Transforme le nombre du scénario en effectif et ajoute une période absente | Produit l’affirmation fausse au lieu de s’abstenir |
| Citation | Identifiant présent, source non pertinente pour l’affirmation | La forme de la citation masque l’absence de preuve |

## Comparaison avec q15 et ablations BM25

Q15 est l’autre question sans réponse attendue. BM25 lui fournit surtout des passages d’anciennes annales, sans calendrier du prochain examen. Son premier score (`10,013`) est presque identique à celui de q16 (`9,875`), mais le modèle s’abstient correctement. Le score brut seul ne distingue donc pas les deux comportements : q16 échoue parce que son contexte contient un nombre plausible, même si ce nombre désigne autre chose.

J’ai comparé trois variantes de recherche sur les 16 questions. Le rappel ci-dessous est calculé au niveau du `source_id` parmi les 14 questions répondables ; c’est une mesure de diagnostic, moins stricte que le rappel par emplacement évalué dans le rapport BM25.

| Variante | Score top 1 q15 | Score top 1 q16 | Plus faible score top 1 répondable | Rappel source @1 / @3 / @5 | MRR source |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 texte + métadonnées | 10,013 | 9,875 | 16,420 | 0,857 / 1,000 / 1,000 | 0,929 |
| BM25 texte seulement | 7,915 | 11,115 | 14,763 | 0,714 / 0,929 / 0,929 | 0,810 |
| Métadonnées + mots fréquents retirés de la requête | 7,876 | 6,953 | 12,556 | 0,786 / 1,000 / 1,000 | 0,881 |

Retirer `a`, `d`, `en` et `l` de la requête fait baisser le score de q16, mais le passage de la machine à café reste premier. Retirer toutes les métadonnées ne règle pas le problème non plus : le même passage reste premier et son score monte, tandis que le rappel des questions répondables baisse. Les métadonnées aident donc certaines requêtes, même si leur poids actuel peut contribuer au bruit.

J’ai aussi séparé l’index texte de l’index des métadonnées, puis combiné les scores `BM25_texte + α × BM25_métadonnées`. Cette mesure utilise les références page/section du protocole, contrairement au tableau précédent qui compte seulement les sources. Toutes les variantes laissent le même passage en première position pour q16.

| Variante | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| BM25 actuel, texte et métadonnées concaténés | 0,800 | 0,933 | 1,000 | 0,929 |
| Texte + `0,25 ×` métadonnées | 0,733 | 0,867 | 0,933 | 0,875 |
| Texte + `0,50 ×` métadonnées | 0,733 | 0,933 | 0,933 | 0,893 |
| Texte + `1,00 ×` métadonnées, index séparés | 0,867 | 0,933 | 0,933 | 0,964 |
| Texte seulement | 0,667 | 0,867 | 0,867 | 0,814 |

Le poids égal améliore ici Recall@1 et MRR, mais perd une référence dans le top 5. Les coefficients sont exploratoires et évalués sur le même petit jeu ; ils ne justifient pas encore de remplacer la recherche actuelle.

Enfin, limiter à un chunk par source produit cinq sources distinctes dans le contexte de q16, contre quatre auparavant. Sur les questions répondables, cette déduplication garde Recall@1 (`0,800`), porte Recall@3 de `0,933` à `1,000`, garde Recall@5 (`1,000`) et conserve le MRR (`0,929`). Elle enlève bien le chunk voisin redondant, mais le premier passage reste le même. Avec ce contexte dédupliqué, `qwen3.5:4b` produit exactement la même réponse qu’avec le contexte initial : il conserve le nombre distracteur et ne s’abstient pas. La redondance contribue donc au bruit, mais n’explique pas l’erreur de q16.

Dans ces trois configurations, les deux scores négatifs restent sous le plus faible score des questions répondables. Un seuil choisi après coup séparerait ce jeu précis. Ce résultat ne suffit pas à calibrer un seuil fiable : il n’y a que deux négatifs, les valeurs changent avec la configuration et un distracteur numérique peut faire échouer le modèle même lorsque le score est similaire à celui d’un autre cas où il s’abstient.

## Test exploratoire d’un vérificateur par modèle

J’ai testé localement `qwen3.5:4b` comme vérificateur, avec température `0` et raisonnement désactivé. Ce sont des essais ponctuels sur les réponses et passages déjà enregistrés ; ils n’ont pas modifié la fiche d’évaluation.

Un premier prompt binaire demandait seulement si le contexte pouvait répondre à la question. Il a rejeté Q15 et Q16, comme attendu, mais aussi Q06, dont les deux faits et les citations avaient reçu la note maximale dans la revue humaine. Cette forme de vérification est donc trop stricte pour servir de filtre de réponse.

Un prompt qui examinait la réponse candidate a ensuite donné des résultats encourageants sur un petit sous-ensemble : Q15 était classée comme abstention, Q16 comme non étayée et Q06 comme étayée. Il a néanmoins classé Q01 comme entièrement étayée, alors que la revue humaine jugeait la réponse partielle.

J’ai alors appliqué une consigne généralisée aux 16 cas. Pour comparer les catégories, j’ai dérivé une étiquette à partir des notes existantes : `grounded` exige une couverture complète et une fidélité notée 2 ; `partial` couvre les réponses incomplètes ou avec une réserve mineure ; `unsupported` correspond à une fidélité notée 0 ; `abstention` correspond à une abstention attendue et réussie. Résultat : **7 catégories correctes sur 16**.

| Catégorie issue de la revue humaine | Nombre | Verdict du vérificateur généralisé |
| --- | ---: | --- |
| Entièrement étayée | 6 | 6 « grounded » |
| Partielle | 6 | 6 « grounded » |
| Non étayée | 3 | 3 « grounded » |
| Abstention correcte | 1 | 1 « abstention » |

Le vérificateur a donc appelé « grounded » les 9 réponses partielles ou non étayées, y compris q16 dans ce passage à l’échelle. Le prompt ciblé avait correctement repéré q16, mais la consigne généralisée ne l’a pas fait. Ce contraste montre qu’un succès sur le seul exemple q16 ne suffit pas à valider le mécanisme. Sur ce jeu pilote, le modèle est utile pour explorer des prompts, mais pas comme arbitre automatique de la fidélité.

## Comparaison contrôlée 4B / 9B

Pour isoler la génération, j’ai comparé `qwen3.5:4b` et `qwen3.5:9b` avec le même script, le même prompt, les mêmes paramètres (`temperature=0`, raisonnement désactivé, `num_predict=384`) et les mêmes passages récupérés par BM25 (`top_k=5`). Les identifiants de chunks sont identiques sur les 16 questions. J’ai aussi relancé le 4B : ses 16 réponses et ses 16 contextes sont identiques à la baseline enregistrée.

Le tag `qwen3.5:9b` correspond ici à un modèle de 9,7 milliards de paramètres quantifié en Q4_K_M ; Ollama annonce un téléchargement de 6,6 Go ([fiche du modèle](https://ollama.com/library/qwen3.5)). Les deux lots ont pris environ 3 min 24 s pour le 4B et 6 min 29 s pour le 9B sur cette machine, soit environ 1,9 fois plus longtemps pour le 9B. Ce sont des durées observées sur un seul passage, chargement du modèle compris, pas un benchmark répété.

| Question | 4B | 9B | Observation |
| --- | --- | --- | --- |
| q15, abstention attendue | Abstention correcte | Abstention correcte | Aucun changement |
| q16, abstention attendue | Reprend le distracteur numérique | Reprend le même distracteur numérique | Le 9B ne corrige pas l’erreur centrale malgré un contexte et un prompt identiques |
| q04, répondable | Réponse donnée, mais une affirmation centrale contredit le contexte | Abstention | Plus prudent, mais ne répond pas alors que le corpus contient la réponse |

Les erreurs de couverture ne disparaissent pas : le 9B omet encore la définition de l’agent dans q01 et un des deux algorithmes attendus dans q13. Une notation exploratoire des faits attendus donne `0,756` de couverture moyenne pour le 9B, contre `0,798` dans la revue humaine du 4B. La notation du 9B n’a pas encore été revue indépendamment ; l’écart est indicatif et ne permet pas de conclure que le 9B est globalement moins bon.

Ce test écarte l’idée que la taille seule suffirait à régler q16. Il confirme une interaction entre le passage distracteur remonté par BM25 et la tendance du modèle à prendre une mention numérique pour une preuve de la relation demandée. Le 9B est plus abstentionniste dans q04, mais il ne refuse toujours pas le distracteur de q16.

## Prototype d’une barrière de preuve

J’ai ajouté un évaluateur expérimental qui demande au modèle de décider si les passages suffisent, élément par élément, et d’associer chaque élément à une citation et à un extrait copié. Le script vérifie ensuite que la citation renvoie à un passage fourni et que l’extrait y figure, en tolérant les différences d’accents, d’espaces et de ponctuation. Il ne change pas le pipeline de réponse. L’évaluation compare les décisions à un **proxy de couverture du contexte** : chaque point attendu doit avoir au moins une de ses références de page ou de section dans le top 5. Ce proxy contrôle la présence de l’emplacement attendu, pas l’entaillement sémantique.

| Vérificateur | Exactitude vs proxy | Cas proxy négatifs acceptés à tort | Cas proxy positifs refusés | Décisions « supported » avec extrait vérifié | Durée du lot |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3.5:4b` | 0,250 | 0 / 3 | 13 / 13 | 0 / 1 | 2 min 23 s |
| `qwen3.5:9b` | 0,375 | 0 / 3 | 10 / 13 | 3 / 3 | 6 min 19 s |

Les deux modèles ont classé q16 comme `unsupported`, mais le 4B rejette tous les contextes qu’il juge devoir accepter et le 9B en rejette dix sur treize. Le 9B a aussi produit trois objets JSON invalides. Le contrôle des extraits confirme leur présence textuelle, sans prouver qu’ils appuient réellement l’élément associé. Cette barrière est trop abstentionniste pour être activée dans le pipeline.

## Seuil BM25 comme garde-fou exploratoire

J’ai comparé les scores du premier résultat pour les 14 questions répondables et les deux questions sans réponse, sans appeler de modèle. Avec la configuration BM25 actuelle (texte et métadonnées concaténés, `k1=1.5`, `b=0.75`), le plus faible score répondable est `16,420` et le plus élevé parmi les deux négatifs est `10,013` (`q16` vaut `9,875`). Un seuil de `12` sépare donc correctement les 16 cas de ce jeu pilote.

Cette séparation dépend de la configuration. Avec l’index texte séparé et `0,25 ×` le score des métadonnées, q16 monte à `12,157` et passerait un seuil fixe de `12`. Un seuil recalé après coup sépare les cas de chaque variante testée, mais il serait ajusté sur les mêmes 16 questions. Deux négatifs ne suffisent pas à démontrer qu’un tel seuil généralisera à des requêtes absentes du jeu ou à un corpus modifié.

Les scores par variante et les matrices de confusion sont détaillés dans le [rapport exploratoire sur le seuil BM25](v6-abstention-threshold.md).

J’ai ajouté `--min-bm25-score` comme option **désactivée par défaut** dans la commande `ask` et dans l’évaluateur de génération. Pour la configuration actuelle, `12` est un candidat à évaluer, pas une valeur calibrée pour un usage général. Le chemin de la commande `ask` a été vérifié sur q16 : il renvoie l’abstention exacte avant l’appel au modèle. Le seuil ne vérifie pas les relations d’une réponse quand le score dépasse la limite, comme le montre notamment l’erreur de q04.

## Pistes de correction

1. **Rendre la décision d’abstention fondée sur la preuve.** Avant de répondre à une question chiffrée, vérifier que le contexte établit le même sujet, la même relation et, si elle est demandée, la même période. Le passage à 9B ne suffit pas : q16 échoue encore avec le même distracteur.
2. **Utiliser le seuil BM25 seulement comme garde-fou provisoire.** Le seuil `12` arrête les deux négatifs du jeu avec la configuration actuelle, mais un changement de pondération suffit à faire passer q16. Il faut davantage de négatifs difficiles et un jeu distinct pour calibration avant de l’activer par défaut.
3. **Réduire le bruit lexical de BM25.** Le score texte + métadonnées à poids égal est un candidat pour améliorer le rang des sources utiles, mais il perd une référence à `@5` et q16 garde le même premier résultat. La déduplication par source réduit les passages voisins et préserve le rappel sur ce jeu ; elle ne suffit pas à obtenir l’abstention.
4. **Ne pas installer le vérificateur Qwen comme filtre automatique à ce stade.** Les essais montrent trop de faux refus et, pour le 9B, des sorties structurées invalides. Il faut une autre conception et une validation sur un jeu plus large avant de lui confier le blocage des réponses.
5. **Conserver q16 dans l’évaluation et ajouter des négatifs avec nombres distracteurs.** La prochaine comparaison doit vérifier que le système s’abstient sans citation trompeuse, tout en répondant aux questions chiffrées réellement couvertes par le corpus.

## Conclusion

q16 révèle une récupération trop permissive et une génération qui confond présence lexicale et preuve. Passer à 9B ne corrige pas l’erreur. Les variantes BM25 et la déduplication ne retirent pas le distracteur de la première place. Un seuil de score à `12` sépare les 14 cas répondables des deux négatifs avec la configuration actuelle ; c’est le garde-fou provisoire le plus simple pour q16, mais les essais sont trop petits pour l’activer par défaut. Les vérificateurs Qwen testés sont trop conservateurs. Avant une mise en service, il faut enrichir le jeu avec des négatifs difficiles, fixer une calibration indépendante et évaluer un contrôle d’entaillement plus fiable.

Les textes des questions, réponses et passages restent dans les fiches locales ignorées par Git, notamment `data/evaluations/generation_review_qwen35_9b.jsonl`, `data/evaluations/generation_review_qwen35_4b_ab.jsonl` et les fiches du vérificateur `data/evaluations/evidence_gate_qwen35_*.jsonl`, conformément au protocole d’évaluation.
