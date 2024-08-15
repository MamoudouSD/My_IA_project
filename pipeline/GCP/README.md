# Projet: Pipeline de Machine Learning sur Vertex AI

## Description
Ce projet implémente un pipeline complet de machine learning pour l'entraînement, l'évaluation, l'enregistrement, et le déploiement de modèles RNN utilisant TensorFlow sur Google Cloud Platform, spécifiquement Vertex AI. Il automatise le flux de travail d'entraînement et de déploiement pour faciliter les mises à jour et les tests de modèles.

## Caractéristiques
- **Framework**: TensorFlow
- **Type de Modèle**: RNN (Réseaux de Neurones Récurrents)
- **Tâches Impliquées**: Prédiction
- **Plateforme de developpement**: Google Cloud Platform.

## Configuration Initiale
1. Définir les variables d'environnement et les paramètres de projet:
   - `PROJECT_ID`: Identifiant du projet GCP
   - `SERVICE_ACCOUNT`: Compte de service utilisé pour l'exécution
   - `GCS_BUCKET`: Bucket Google Cloud Storage pour stockage des données et modèles

## Composants du pipeline
Le pipeline inclut les étapes suivantes:

Entraînement du modèle:
Utilise train.py pour l'entraînement du modèle sur des données préparées.
Évaluation et validation du modèle:
Compare les performances avec les versions précédentes et valide la qualité avant déploiement.
Déploiement du modèle:
Met à jour le modèle en production en fonction des résultats de l'évaluation.

## Structure du pipeline
![1](https://github.com/user-attachments/assets/fce394e2-c030-4fb4-a33c-4239f5d60920)
![2](https://github.com/user-attachments/assets/a4a4c634-6225-4550-bcb1-7c129a15bde0)
![3](https://github.com/user-attachments/assets/1161e141-9d31-480c-84d5-86ccbf8d6ecf)
