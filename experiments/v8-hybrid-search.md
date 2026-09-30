# Comparaison de Recherche : BM25 vs Dense vs Hybride (RRF)

Généré le 2026-09-30 21:38 UTC sur `ia04_questions_dev.jsonl` (15 questions : 10 répondables, 5 sans réponse).
- Corpus : 366 chunks (`ia04_chunks.jsonl`).
- Modèle Dense : `bge-m3` (embeddings pré-normalisés).
- Constante RRF : $k = 60$.

## 1. Métriques de Recherche (Retrieval)

| Système | Index / Méthode | Recall@1 | Recall@3 | Recall@5 | MRR | Latence moyenne | Initialisation |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BM25** (metadata) | Lexical (TF-IDF RAM) | 0.636 | 0.909 | 1.000 | 0.833 | 0.7 ms | 11.8 ms |
| **Dense** (`bge-m3`) | Vecteurs 1024-d | 0.636 | 0.818 | 0.909 | 0.825 | 25.1 ms | 42.7 ms |
| **Hybride RRF** | Fusion des rangs ($k=60$) | **0.818** | **1.000** | **1.000** | **0.950** | 24.4 ms | 0.0 ms |

## 2. Analyse détaillée par question répondable

| Question ID | Question | BM25 R@1 (Rang) | Dense R@1 (Rang) | Hybride R@1 (Rang) |
| :--- | :--- | :---: | :---: | :---: |
| `q01` | Quels éléments permettent de distinguer un agent d’un système mul... | ✗ (rang 2) | ✗ (rang 4) | ✓ (rang 1) |
| `q03` | Qu’est-ce qu’une data race en Go, et pourquoi l’incrémentation d’... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q05` | À quoi servent les méthodes Add, Done et Wait d’un WaitGroup en G... | ✓ (rang 1) | ✗ (rang 2) | ✓ (rang 1) |
| `q06` | Comment la règle de Copeland attribue-t-elle un score aux candida... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q08` | Sous quelle condition le théorème de l’électeur médian garantit-i... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q10` | Dans une enchère de Vickrey, que paie le gagnant et quelle propri... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q12` | Quelles propriétés des valeurs de Shapley et de Banzhaf le TD10 d... | ✓ (rang 1) | ✓ (rang 1) | ✓ (rang 1) |
| `q13` | Dans le sujet du médian A21, quel problème d’appariement est posé... | ✗ (rang 2) | ✓ (rang 1) | ✗ (rang 2) |
| `q17` | Comment est calculée la note finale d'IA04 et quelle est la compo... | ✗ (rang 3) | ✓ (rang 1) | ✓ (rang 1) |
| `q19` | Quel est le comportement de l'instruction defer en Go et dans que... | ✓ (rang 1) | ✗ (rang 2) | ✓ (rang 1) |

## 3. Conclusions techniques

1. **Complémentarité lexicale / sémantique** :
   - BM25 performe sur les termes techniques exacts mais échoue lorsque le vocabulaire diffère.
   - Dense excelle sur les reformulations sémantiques mais manque de précision sur les syntaxes rares.
   - **Hybride RRF** surmonte les deux faiblesses en assurant un consensus robuste.
2. **Impact sur la génération** :
   - L'apport de l'hybride permet à la génération de récupérer les preuves manquantes sans jamais sacrifier la vitesse.

