<div align="center">

![Bannière du projet — Airbnb NYC 2019](assets/banner.svg)

# Airbnb NYC 2019 - Analyse et prédiction des prix

**Nettoyage des données · Analyse exploratoire · Régression supervisée**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)](https://seaborn.pydata.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine_Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

**Projet académique - Domaine : Tourisme**  
**Réalisé par : Vivien AGBO & ADANLAO Laurinda**

</div>

---

## Présentation du projet

Ce projet porte sur l'exploration du marché des hébergements Airbnb à New York à partir du jeu de données **New York City Airbnb Open Data (2019)**. Il s'inscrit dans une démarche de Data Science appliquée au domaine du tourisme.

Le travail suit trois axes complémentaires :

1. **Nettoyer les données** afin d'identifier les problèmes de qualité, de corriger les incohérences justifiées et de conserver une trace des décisions prises.
2. **Explorer les annonces** afin de comprendre la répartition de l'offre, les différences de prix entre arrondissements et types d'hébergement, ainsi que les caractéristiques du marché.
3. **Réaliser un premier essai de modélisation** pour estimer le prix affiché par nuit à partir des caractéristiques d'une annonce.

> **Périmètre important :** le jeu de données décrit des annonces new-yorkaises en 2019. Les résultats sont donc historiques et ne doivent pas être interprétés comme une photographie du marché Airbnb actuel.

## Problématique

**Dans quelle mesure les caractéristiques géographiques, le type d'hébergement et les informations associées aux annonces permettent-ils de comprendre les tarifs Airbnb à New York et d'en proposer une estimation ?**

Cette question est abordée en deux temps : une analyse descriptive du marché, puis un problème de régression supervisée dont la variable cible est `price`.

## Jeu de données

| Élément | Description |
|---|---|
| Nom | New York City Airbnb Open Data |
| Fichier | `AB_NYC_2019.csv` |
| Période | 2019 |
| Source de téléchargement | [Kaggle - New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) |
| Nombre initial d'annonces | 48 895 |
| Nombre de variables | 16 |
| Format | CSV |
| Zone géographique | Les cinq grands arrondissements de New York |

Le dataset contient notamment les identifiants des annonces et des hôtes, les arrondissements et quartiers, les coordonnées géographiques, le type de logement, le prix par nuit, le nombre minimum de nuits, les avis et la disponibilité sur 365 jours.

### Principales variables

| Variable | Signification |
|---|---|
| `neighbourhood_group` | Grand arrondissement de New York |
| `neighbourhood` | Quartier de l'annonce |
| `latitude`, `longitude` | Coordonnées géographiques |
| `room_type` | Type d'hébergement proposé |
| `price` | Prix affiché par nuit, en USD |
| `minimum_nights` | Nombre minimum de nuits exigées |
| `number_of_reviews` | Nombre total d'avis reçus |
| `reviews_per_month` | Moyenne d'avis par mois |
| `availability_365` | Nombre de jours disponibles à la réservation sur 365 jours |

Le CSV n'est pas inclus dans le dépôt. Le notebook tente de le télécharger automatiquement depuis Kaggle. Il est également possible de le placer manuellement dans `data/raw/`.

## Résultats clés du travail

Les indicateurs ci-dessous proviennent de l'exécution du notebook sur le fichier attribué au projet.

### Nettoyage des données

| Indicateur | Résultat |
|---|---:|
| Lignes initiales | 48 895 |
| Lignes après nettoyage | 48 884 |
| Annonces écartées | 11 |
| Part des annonces conservées | 99,98 % |
| Cellules manquantes avant nettoyage | 20 141 |
| Cellules manquantes après nettoyage | 20 102 |

Les 11 annonces écartées présentaient un prix nul ou négatif, non exploitable pour l'analyse tarifaire. Aucun doublon complet ni identifiant d'annonce répété n'a été détecté. Les valeurs de prix élevées et les durées minimales atypiques ont été conservées lorsqu'elles ne constituaient pas à elles seules une preuve d'erreur.

Les absences dans `last_review` et `reviews_per_month` ont également été examinées selon le nombre d'avis : une annonce n'ayant reçu aucun avis peut légitimement ne pas avoir de date de dernier avis ni de moyenne mensuelle.

### Principaux constats de l'analyse exploratoire

- **Concentration géographique :** Manhattan compte 21 661 annonces et Brooklyn 20 104 dans le fichier initial. Ensemble, ces deux arrondissements représentent environ 85,42 % des annonces.
- **Type d'hébergement :** les logements entiers sont majoritaires, avec 25 409 annonces initiales, soit environ 51,97 %.
- **Prix :** le prix médian observé est de 106 USD par nuit, tandis que la moyenne atteint 152,76 USD. L'écart suggère une distribution asymétrique, influencée par des annonces plus chères.
- **Manhattan :** le prix médian y atteint 150 USD par nuit dans l'analyse réalisée.

Ces statistiques sont descriptives. Elles ne démontrent pas, à elles seules, qu'un facteur cause un prix plus élevé.

### Essai de modélisation - régression

La cible est `price`. Les données ont été séparées en un ensemble d'entraînement (80 %) et un ensemble de test (20 %). Les deux modèles apprennent sur `log(1 + price)`, puis les prédictions sont reconverties en dollars pour l'évaluation.

Les approches comparées sont :

- une **baseline** qui prédit systématiquement la médiane des prix de l'entraînement ;
- une **régression Ridge**, modèle linéaire régularisé ;
- une **forêt aléatoire de régression**, capable de modéliser des relations non linéaires.

| Modèle | MAE sur le jeu de test |
|---|---:|
| Baseline - médiane | 82,73 USD |
| Régression Ridge | 59,72 USD |
| Forêt aléatoire | **55,49 USD** |

La forêt aléatoire obtient la MAE la plus faible parmi les trois approches testées. Son coefficient de détermination est toutefois de **R² = 0,203** : le modèle reste perfectible et ne doit pas être présenté comme un outil de tarification prêt à être déployé.

> Les performances correspondent à une séparation entraînement/test donnée et à cette version historique du dataset. Elles ne garantissent pas les mêmes résultats sur d'autres échantillons ou sur des annonces actuelles.

## Méthodologie

Le notebook suit un workflow organisé en étapes :

1. **Compréhension métier** : définition du contexte touristique, de la question et de la cible.
2. **Chargement** : import du CSV et conservation d'une copie brute.
3. **Diagnostic initial** : dimensions, types, valeurs manquantes, doublons, incohérences et valeurs atypiques.
4. **Nettoyage** : traitement des prix non positifs, harmonisation des textes, conversion des types et contrôles de cohérence.
5. **Évaluation du nettoyage** : comparaison des indicateurs avant et après traitement.
6. **EDA** : répartition géographique, types d'hébergement, distribution et comparaison des prix, séjours minimums, corrélations et carte.
7. **Préparation ML** : sélection des variables, création de `has_reviews`, imputation, encodage et standardisation lorsque nécessaire.
8. **Modélisation** : entraînement de Ridge et Random Forest, comparés à une baseline.
9. **Évaluation** : MAE, RMSE, MedAE et R², graphiques des prédictions et analyse des variables importantes.
10. **Export** : vérifications finales et génération du CSV nettoyé.

Les traitements de préprocessing utilisés par les modèles sont intégrés dans des pipelines scikit-learn. Ils sont ajustés sur l'échantillon d'entraînement afin de limiter le risque de fuite de données (*data leakage*).

## Structure du dépôt

```text
airbnb-nyc-tourism-analysis/
├── assets/
│   └── banner.svg
├── data/
│   ├── raw/                 # CSV source (non versionné)
│   ├── processed/           # CSV nettoyé (non versionné)
│   └── README.md
├── notebooks/
│   └── 01_airbnb_nyc_cleaning_eda_modeling.ipynb
├── reports/
│   ├── Presentation_Airbnb_Interpretee.pdf
│   └── Presentation_Airbnb_Interpretee.pptx
├── .gitignore
├── README.md
└── requirements.txt
```

Une démonstration interactive réalisée avec **Streamlit** sera ajoutée dans une prochaine étape du projet.

## Installation et exécution

### 1. Cloner le dépôt

Remplacez `<votre-utilisateur>` par votre nom d'utilisateur GitHub :

```bash
git clone https://github.com/<votre-utilisateur>/airbnb-nyc-tourism-analysis.git
cd airbnb-nyc-tourism-analysis
```

### 2. Créer un environnement virtuel

Sous Ubuntu/Linux :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Sous Windows PowerShell :

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Récupérer les données

Ouvrez le notebook dans JupyterLab ou VS Code. Le notebook cherche d'abord `data/raw/AB_NYC_2019.csv`, puis tente un téléchargement via KaggleHub et, en solution de secours, via l'API Kaggle.

Si l'API Kaggle demande une authentification, configurez votre jeton Kaggle localement. **Ne commitez jamais `kaggle.json`**. Vous pouvez aussi télécharger manuellement le CSV depuis la page Kaggle et le déposer dans `data/raw/`.

### 5. Lancer le notebook

```bash
jupyter lab
```

Ouvrez ensuite `notebooks/01_airbnb_nyc_cleaning_eda_modeling.ipynb` et exécutez les cellules dans l'ordre.

Le notebook exporte le fichier nettoyé sous `data/processed/AB_NYC_2019_cleaned.csv`.

## Livrables

- Notebook Python commenté en français avec diagnostic, nettoyage, interprétations, EDA et essai de modélisation.
- CSV nettoyé généré localement à l'exécution.
- Présentation PowerPoint modifiable et version PDF dans `reports/`.
- Application Streamlit interactive : à venir.

## Limites et précautions d'interprétation

- Les données décrivent les annonces de 2019 ; elles ne reflètent pas nécessairement le marché actuel.
- `price` correspond au prix affiché par nuit dans le dataset, pas obligatoirement au montant final payé par un voyageur.
- `availability_365` correspond à une disponibilité déclarée dans le dataset et ne doit pas être assimilée automatiquement à un taux d'occupation réel.
- Les corrélations et importances de variables ne prouvent pas une relation causale.
- Le R² de la forêt aléatoire (0,203) indique que le modèle explique une part limitée de la variabilité du prix. Il s'agit d'un essai pédagogique, pas d'un système de tarification prêt pour la production.

## Références et attribution

- **Jeu de données utilisé :** [New York City Airbnb Open Data — Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).
- **Projet Inside Airbnb :** [insideairbnb.com](https://insideairbnb.com/).
- **Documentation Python :** [python.org](https://www.python.org/).
- **Documentation Pandas :** [pandas.pydata.org](https://pandas.pydata.org/).
- **Documentation Seaborn :** [seaborn.pydata.org](https://seaborn.pydata.org/).
- **Documentation scikit-learn :** [scikit-learn.org](https://scikit-learn.org/).

La fiche Kaggle du jeu de données indique une licence **CC0 — Public Domain** pour cette version. Vérifiez toujours la licence et les obligations d'attribution de la source exacte avant de redistribuer un fichier de données.

---

<div align="center">

**Réalisé dans le cadre d'un travail pratique de Data Science — Domaine Tourisme**  
*Du nettoyage des données à un premier modèle de prédiction des prix.*

</div>
