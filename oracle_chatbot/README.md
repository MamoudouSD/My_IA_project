# Projet: Chatbot Basé sur LangChain et FAISS

## Description

Ce projet met en place un chatbot intelligent qui utilise LangChain pour la génération de texte et FAISS pour la recherche de similarité. Le chatbot est capable de récupérer et de répondre à des requêtes utilisateur en se basant sur des documents indexés au préalable. Il est conçu pour être déployé sur une plateforme web en utilisant Streamlit, offrant ainsi une interface utilisateur simple et interactive.

## Caractéristiques

- **Framework**: LangChain
- **Moteur de Recherche**: FAISS (Facebook AI Similarity Search)
- **Tâches Impliquées**: Recherche de similarité, Génération de texte, Chatbot conversationnel
- **Plateforme de Déploiement**: Streamlit, Oracle Cloud Infrastructure (OCI)

## Configuration Initiale

Avant de démarrer, assurez-vous de configurer correctement les variables d'environnement et les paramètres nécessaires:

- **LANGCHAIN_API_KEY**: Clé API pour l'accès aux services de LangChain.
- **COMPARTMENT_ID**: Identifiant du compartiment OCI utilisé pour l'exécution.
- **SERVICE_ENDPOINT**: URL de l'endpoint du service OCI utilisé pour la génération de texte et l'embedding.

## Composants du Projet

Le projet se compose des étapes suivantes:

1. **Création et Indexation des Documents avec FAISS**: 
   - Utilisez le script `faiss_create.py` pour charger, segmenter, et indexer vos documents en utilisant FAISS. Cette étape est cruciale pour préparer le chatbot à effectuer des recherches de similarité efficaces sur les documents fournis.

2. **Déploiement du Chatbot**: 
   - Le script `Chatbot.py` est utilisé pour lancer le chatbot sur un serveur local via Streamlit. Il intègre les capacités de LangChain pour la génération de réponses basées sur les requêtes utilisateurs et les documents indexés.

3. **Utilisation du Chatbot**:
   - Une fois le serveur démarré, le chatbot est accessible via un navigateur web. Il est capable de répondre en temps réel en utilisant les modèles de génération de texte configurés.

## Informations Complémentaires

Pour plus de détails sur la configuration et l'utilisation, consultez le fichier `procedure.pdf`. Notez que ce fichier peut nécessiter un logiciel spécifique pour être ouvert.
