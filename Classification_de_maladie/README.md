Les maladies cardiovasculaires, constituant une préoccupation majeure de santé publique, exigent des avancées significatives dans les méthodes de diagnostic pour permettre une intervention précoce et efficace. Les codes ci-dessus ont été utilisés dans le cadre d'une étude de maîtrise en recherche. Cette étude s'inscrit dans le cadre de l'amélioration des capacités diagnostiques des maladies cardiovasculaires en explorant l'utilisation novatrice de la décomposition en ondelettes discrètes (DWT) pour optimiser la performance des réseaux neuronaux dans la classification de ces pathologies.

L'étude utilise la base de données ECG PTB-XL de PhysioNet pour prédire les maladies cardiovasculaires. Cette base de données comprend 21 799 enregistrements d'ECG de 18 869 patients et est largement utilisée dans la recherche sur l'analyse ECG et l'apprentissage automatique. Les données sont regroupées en cinq catégories, notamment ECG normal, Infarctus du myocarde, Changement ST-T, Trouble de la conduction, et Hypertrophie.

Pour la préparation des données, les enregistrements sont catégorisés en Cross-validation Folds de 1 à 10, avec la recommandation d'utiliser les plis de 1 à 8 pour l'entraînement, le pli 9 pour la validation, et le pli 10 pour le test. Les signaux ECG sont transformés en coefficients DWT (Transformée en ondelettes discrète) pour capturer les caractéristiques pertinentes.

La normalisation des données est effectuée pour les mettre sur une même échelle. Un modèle de réseau de neurones convolutifs (CNN) est utilisé pour la classification des maladies cardiovasculaires. Le modèle comprend une phase d'extraction des caractéristiques et une phase de classification avec des couches de convolution, d'activation ReLU, et de normalisation par lots. La précision, la précision, le rappel, et le score F1 sont utilisés comme métriques d'évaluation.

Les résultats montrent que l'approche avec la décomposition DWT (Db4 et Haar) surpasse les modèles utilisant des données brutes et des architectures telles que ResNet50, VGG16, et Inception-ResNet-v2 en termes de précision, de précision, de rappel, et de score F1. Les modèles DWT obtiennent des résultats compétitifs avec des paramètres d'entraînement optimisés via Hyperopt, mettant en évidence l'efficacité de cette approche pour la classification des maladies cardiovasculaires.

# Structure et Référence de code
  - Exemple d'optimisation : C'est un code qui teste different hyper parametre du modele
  - InvRes_classification_maladie : Ce code concoit le modele Inversio Resnet avec des convolutions a une dimension (DOI: https://doi.org/10.1609/aaai.v31i1.11231)
  - Resnet50_classification_maladie : Ce code concoit le modele Resnet 50 avec des convolutions a une dimension (https://cv-tricks.com/keras/understand-implement-resnets/)
  - VGG16_classification_maladie : Ce code concoit le modele VGG16 avec des convolutions a une dimension (https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918)
  - Mon_model_classification_maladie : Ce code concoit mon modele de classification avec des convolutions a une dimension.
