# Données du projet

Les fichiers de données ne sont pas versionnés dans Git afin de garder le dépôt léger et d'éviter de redistribuer le jeu de données source.

- `raw/` : placer ici le fichier original `AB_NYC_2019.csv`.
- `processed/` : le notebook y exporte `AB_NYC_2019_cleaned.csv` après exécution.

Le notebook tente également de télécharger automatiquement le jeu de données depuis Kaggle. En cas de demande d'authentification, configurez vos identifiants Kaggle localement et ne les ajoutez jamais au dépôt.
