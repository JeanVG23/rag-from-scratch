# Inventaire et audit initial du corpus IA04

Audit réalisé le 24 septembre 2026. Seul `data/raw/IA04/` est inclus ; les autres cours présents localement sont hors périmètre.

## Inventaire

- 20 documents : 3 supports de cours Markdown, 11 sujets de TD Markdown et 6 annales PDF.
- Les sources Markdown sont lisibles comme texte et contiennent des titres de sections utiles pour les références.
- L’inventaire local et les empreintes SHA-256 par fichier sont enregistrés dans `data/manifest.jsonl`, ignoré par Git. L’empreinte agrégée de cet état du corpus est `634cd8d5dd464b94b71ac537d47dc9b796665c36a5f40386ca419c0cb36bb0bf`.

## Vérification des PDF

Les six PDF contiennent 23 pages au total. Un contrôle avec `pdftotext -bbox-layout` (Poppler 26.08.0) retrouve les 23 pages annoncées par `pdfinfo`. Aucune page n’est vide ; les 7 863 blocs `<word>` ont des coordonnées valides dans leur page. Cela confirme la présence d’une couche texte, mais pas l’exactitude de chaque formule, tableau ou ordre de lecture.

Le contrôle visuel de la première page du médian A21 confirme que l’ordre natif de `pdftotext` n’est pas fiable : il place les questions 4 à 6 de la colonne droite avant le contenu de la colonne gauche. `-layout` mélange également les lignes des colonnes. En regroupant les lignes avec leurs coordonnées, puis en lisant la colonne gauche de haut en bas avant la droite, on retrouve l’ordre imprimé sur cette page. Des contrôles visuels des pages 1 et 2 de l’examen final A21 montrent aussi des colonnes et des tableaux qui demandent de conserver les zones séparées.

Un tri générique par abscisse n’est cependant pas suffisant. Il prend notamment la page 7 de l’examen A24 — qui contient deux matrices larges, et non deux colonnes de texte — pour une page à colonnes ; le tri casse alors l’ordre naturel des cellules. Les formules et tableaux doivent donc être conservés comme des blocs ou traités par une règle adaptée. L’ordre de lecture géométrique n’est pas encore validé sur chaque page IA04.

Un essai distinct a porté sur la première page de `Cours1_LO23_4pp.pdf`, qui contient quatre diapositives. `pdftotext` extrait bien le texte des quatre zones, mais l’ordre linéaire mélange le titre de la première diapositive avec le contenu de la deuxième. En regroupant les lignes par quadrants à partir de leurs coordonnées, puis en les lisant haut-gauche, haut-droite, bas-gauche et bas-droite, on retrouve les quatre diapositives dans l’ordre.

Deux fichiers Markdown contiennent une image distante ; dans le sujet de TD9, l’arbre de jeu est fourni sous forme d’image externe et ne figure pas dans le texte Markdown. Il faudra garder cette limite à l’esprit pour les questions de théorie des jeux.

## Décision pour la suite

L’ingestion utilisera `pdftotext -bbox-layout` pour disposer du texte et de ses coordonnées. Le regroupement géométrique doit distinguer les colonnes de texte des tableaux et matrices ; il reste à implémenter et à vérifier sur tout IA04 avant de considérer l’ordre d’extraction comme validé. Les références garderont le numéro de page d’origine et, pour une page à plusieurs diapositives, l’identifiant de la zone. Les PDF LO23 restent hors du corpus initial.
