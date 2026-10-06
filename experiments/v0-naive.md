# Expérience v0 : Baseline Naïve par recouvrement de mots

Première référence historique du projet avant l'implémentation de BM25 et des embeddings.

## 1. Méthode

- **Modèle** : `NaiveOverlapRetriever` ([src/rag_from_scratch/retrieval/naive.py](../src/rag_from_scratch/retrieval/naive.py)).
- **Principe** : Comptage direct du nombre de mots communs entre la requête et le passage brut ($|\text{tokens}(q) \cap \text{tokens}(d)|$).
- **Limites conceptuelles identifiées** :
  1. Pas de pondération **IDF** : les mots fréquents et peu informatifs ont le même poids que les termes techniques rares.
  2. Pas de normalisation par la longueur : les longs passages sont artificiellement favorisés.
  3. Répétitions non prises en compte.

## 2. Enseignements

Cette baseline a motivé :
- L'implémentation d'un vrai moteur **Okapi BM25** avec normalisation de longueur et pénalisation des termes fréquents ([v2](v2-bm25.md)).
- L'injection des métadonnées de structure (titres de sections et noms de fichiers) dans l'index ([v3](v3-bm25-metadata.md)).
