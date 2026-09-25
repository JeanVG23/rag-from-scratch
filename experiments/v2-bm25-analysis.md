# Analyse des erreurs BM25 IA04

Analyse générée le 2026-09-25 09:53 UTC.
- Corpus : 366 chunks depuis `data/gold/ia04_chunks.jsonl`.
- Questions : 14 répondables sur 16 ; même jeu et mêmes critères que `v2-bm25.md`.
- Les variantes sont évaluées sur les mêmes questions et les mêmes références ; la baseline BM25 n’est pas modifiée.
- Les textes de questions et des cours ne sont pas recopiés dans ce rapport.

## Diagnostic ciblé

| Question | Référence attendue | Rang BM25 initial | Observation |
| --- | --- | ---: | --- |
| `q01` | `course_03_communication_multi_agents`, puis `course_00_intro` | 2 puis 8 | La recherche trouve une des deux références dans le top 5 ; l’autre section reste au rang 8. |
| `q13` | `annales_a21_median_ia04`, page 1 | 17 | Les meilleurs passages lexicaux parlent d’autres annales ; la référence attendue n’entre pas dans le top 5. |

L’inspection des termes confirme deux signaux pour `q13` : la question emploie le pluriel « algorithmes » alors que le chunk de référence contient « algorithme » ; l’identifiant `A21` figure dans le nom du fichier source, absent du texte scoré. Le chunk de référence contient aussi le terme « médian ». Les mots `appariement`, `A21` et `médian` apparaissent chacun dans plusieurs chunks, tandis que les termes fréquents contribuent encore au score.

Pour `q01`, les deux sources attendues sont liées à des titres de section. Le retriever ne scorait jusque-là que `chunk.text`, sans utiliser `chunk.section` ni le nom du fichier source.

## Ablation contrôlée

| Variante | Recall@1 | Recall@3 | Recall@5 | MRR | Rang ref. `q01` | Recall@5 `q01` | Rang ref. `q13` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Texte seul (baseline) | 0.667 | 0.867 | 0.867 | 0.814 | 2 | 0.500 | 17 |
| Normalisation légère du pluriel | 0.533 | 0.867 | 1.000 | 0.752 | 2 | 1.000 | 3 |
| Nom de fichier | 0.800 | 0.933 | 0.933 | 0.929 | 2 | 0.500 | 1 |
| Titre de section | 0.800 | 0.867 | 0.933 | 0.897 | 2 | 1.000 | 18 |
| Nom de fichier + titre de section | 0.800 | 0.933 | 1.000 | 0.929 | 2 | 1.000 | 2 |
| Nom, section et pluriel | 0.667 | 0.933 | 0.933 | 0.845 | 3 | 0.500 | 1 |

La variante nom de fichier + titre de section retrouve les deux références de `q01` dans le top 5 et fait passer la référence de `q13` du rang 17 au rang 2. La normalisation plurielle seule fait progresser `q13`, mais diminue le MRR global ; elle n’est donc pas retenue comme correction générale.

## Décision

Le meilleur candidat de cette analyse est l’ajout du nom de fichier et du titre de section à la recherche : Recall@1 0.800, Recall@3 0.933, Recall@5 1.000, MRR 0.929.
Ces résultats sont exploratoires : les mêmes 14 questions ont servi à diagnostiquer et à comparer les variantes. Ils motivent une version BM25 avec métadonnées, sans établir une performance générale. La normalisation morphologique reste une piste séparée.

Détails sans textes de questions ni de chunks : `data/evaluations/bm25_ablation.json`.
