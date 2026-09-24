# Comparaison des embeddings IA04

Évaluation générée le 2026-09-24 12:56 UTC.
- Chunks : 366 depuis `data/gold/ia04_chunks.jsonl`.
- Questions répondables : 14 sur 16 ; les questions sans réponse sont exclues, conformément au protocole.
- Recherche : similarité cosinus sur les vecteurs normalisés, top-k évalué à 1, 3 et 5.
- Recall@k : proportion micro des références document/section/page retrouvées ; MRR : rang du premier résultat pertinent, moyenné par question.

| Modèle Ollama | Dimension | Recall@1 | Recall@3 | Recall@5 | MRR | Embedding corpus (s) | Embedding questions (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bge-m3` | 1024 | 0.667 | 0.733 | 0.867 | 0.791 | 19.73 | 0.29 |
| `qwen3-embedding:0.6b` | 1024 | 0.533 | 0.867 | 1.000 | 0.740 | 32.86 | 0.45 |
| `corvojaeger/jina-embeddings-v5-text-small` | 1024 | 0.733 | 0.733 | 1.000 | 0.836 | 34.39 | 0.43 |

## Versions Ollama

- `bge-m3` : `bge-m3:latest`, digest `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`, 1.08 Gio téléchargés.
- `qwen3-embedding:0.6b` : `qwen3-embedding:0.6b`, digest `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`, 0.60 Gio téléchargés.
- `corvojaeger/jina-embeddings-v5-text-small` : `corvojaeger/jina-embeddings-v5-text-small:latest`, digest `953c50af9cca68235b09014cb76c5fd51fc7bfa37e73b3eac9964d8423bfe86e`, 0.37 Gio téléchargés.

## Paramètres et limites

- Découpage initial : fenêtres de 400 mots avec 50 mots de chevauchement, sans traverser une page ou une section.
- Qwen3 reçoit une instruction sur les requêtes et les passages restent bruts ; Jina reçoit les préfixes `Query:` et `Document:` ; BGE-M3 reçoit le texte brut.
- Jina v5 est utilisé via un [paquet Ollama communautaire](https://ollama.com/corvojaeger/jina-embeddings-v5-text-small) ; le modèle amont est sous licence [CC BY-NC 4.0](https://jina.ai/en-US/models/jina-embeddings-v5-text-small/), donc ce résultat reste une comparaison locale de projet pédagogique.
- Les scores portent sur le petit jeu pilote privé de 14 questions répondables. Ils donnent un premier signal pour IA04, pas un classement général des modèles.
- L’extraction PDF utilise actuellement `pdftotext -layout`. L’audit a relevé des pages à colonnes et des tableaux/matrices ; leur ordre de lecture n’est pas validé sur toutes les pages, ce qui peut influencer le résultat.
- Les index vectoriels restent locaux dans `data/indexes/` et ne sont pas distribués.

Les classements détaillés par question et les vecteurs sont conservés dans les artefacts locaux sous `data/`.
