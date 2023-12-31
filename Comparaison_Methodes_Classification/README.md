# Introduction
L'essence principale de cette étude résider dans la comparaison et l'analyse des performances de divers algorithmes de classification sur un ensemble de données donné, tout en évaluant l'impact de la sélection d'attributs sur ces performances.

# Algorithmes de Classification Étudiés
Les algorithmes de classification examinés dans le document sont les suivants :

  - Arbre de décision
  - Forêt d’arbres décisionnels (Random Forest)
  - Bagging
  - AdaBoost
  - Classification bayésienne naïve

Ces algorithmes ont été implémentés et comparés pour évaluer leurs performances sur un jeu de données spécifique.

# Métriques d'Évaluation

Les métriques d'évaluation utilisées pour comparer les performances des algorithmes de classification sont les suivantes :

  - Taux de Vrai Positif (TP Rate)
  - Taux de Faux Positif (FP Rate)
  - Aire sous la courbe ROC (Receiver Operating Characteristic)
  - F-mesure

Ces métriques sont employées pour évaluer la capacité des algorithmes à effectuer des prédictions correctes, à minimiser les erreurs de classification et à fournir des mesures de précision et de rappel.

# Algorithmes de sélection

Les algorithmes utilisés pour la sélection d'attributs sont le gain d'information (IG) et le test du chi2. Ces algorithmes sont utilisés pour évaluer l'importance des attributs dans le cadre de la classification.

# Résultats

## Résultats avec tous les attributs

| Modèle            | TP Rate  | FP Rate  | Aire (ROC)  | F-mesure  |
|-------------------|----------|----------|-------------|-----------|
| Arbre de décision | 0.9237988| 0.0901960| 0.9168014   | 0.9233861 |
| Random Forest     | 0.9557541| 0.0588235| 0.9484653   | 0.9536438 |
| Bagging           | 0.9448044| 0.0590849| 0.9428597   | 0.9475810 |
| AdaBoost          | 0.9452513| 0.0674509| 0.9438803   | 0.9389002 |
| Bayésienne naïve  | 0.6538547| 0.0841830| 0.7848358   | 0.7577366 |

## Résultats avec les 7 meilleurs attributs par le Gain d'info

| Modèle            | TP Rate  | FP Rate  | Aire (ROC)  | F-mesure  |
|-------------------|----------|----------|-------------|-----------|
| Arbre de décision | 0.9068156| 0.1043137| 0.9012509   | 0.9086430 |
| Random Forest     | 0.9494972| 0.0758169| 0.9368401   | 0.9427273 |
| Bagging           | 0.9356424| 0.0711111| 0.9322656   | 0.9343603 |
| AdaBoost          | 0.9325139| 0.0747712| 0.9288713   | 0.9341840 |
| Bayésienne naïve  | 0.7707262| 0.1333333| 0.8186964   | 0.8178800 |

## Résultats avec les 7 meilleurs attributs par le Chi2

| Modèle            | TP Rate  | FP Rate  | Aire (ROC)  | F-mesure  |
|-------------------|----------|----------|-------------|-----------|
| Arbre de décision | 0.9097206| 0.0933333| 0.9081936   | 0.9145231 |
| Random Forest     | 0.9459217| 0.0619607| 0.9419805   | 0.9466621 |
| Bagging           | 0.9289385| 0.0624836| 0.9332274   | 0.9415204 |
| AdaBoost          | 0.9275977| 0.0622222| 0.9326877   | 0.9365974 |
| Bayésienne naïve  | 0.7211173| 0.1048366| 0.8081403   | 0.7964951 |


L'analyse des résultats sur les ensembles d'attributs importants révèle que, pour des algorithmes tels que l'arbre de décision, le Random Forest et l'AdaBoost, les attributs identifiés par le test du chi2 sont plus pertinents que ceux du gain d'information, avec une légère variation. En revanche, la classification bayésienne naïve montre une meilleure précision avec les attributs du gain d'information. L'utilisation de l'ensemble complet d'attributs améliore la performance de l'arbre de décision, du Random Forest, du Bagging et de l'AdaBoost, tandis que la classification bayésienne se distingue par des performances supérieures avec moins d'attributs. Ceci souligne l'efficacité de la classification bayésienne dans des contextes où les données sont limitées, montrant des résultats robustes même avec un ensemble restreint d'attributs.
