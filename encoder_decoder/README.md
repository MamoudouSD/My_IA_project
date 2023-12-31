# Introduction

L'objectif principal de ce document était de concevoir et de former une architecture encodeur-décodeur utilisant un réseau de neurones convolutif (CNN) pour la reconstruction et la prédiction d'images. La conception visait à assurer une symétrie entre les structures d'encodeur et de décodeur, garantissant ainsi que l'image reconstruite correspond à l'image originale en termes de taille. 

# Architecture du Modèle

Le modèle développé est un auto-encodeur convolutif composé de 21 couches, intégrant une partie encodeur et une partie décodeur. L'encodeur comprend des couches Conv2D et MaxPooling2D avec des filtres variant en nombre, allant de 512 à 128, et des activations ReLU. De l'autre côté, le décodeur est constitué de couches Conv2D et MaxPooling2D avec une configuration symétrique, utilisant également des filtres et des activations similaires.

L'approche adoptée pour l'élaboration de l'architecture était itérative. Les premiers tests ont révélé que les performances souhaitées n'étaient pas atteintes, conduisant à l'expérimentation avec des augmentations du nombre de filtres et de couches. Cette approche a été inspirée par des architectures similaires disponibles en ligne. Finalement, après plusieurs ajustements, notamment l'augmentation du nombre de couches, le modèle a atteint les résultats escomptés.

# Évaluation du Modèle

Après avoir formé le modèle, il a été évalué en utilisant un classificateur à vecteurs de support linéaire (SVC). Sur le jeu de données d'embedding, l'accuracy obtenue variait autour de 0,775. Cependant, lors de l'application directe du SVC sur les données de test, l'accuracy était légèrement inférieure, oscillant autour de 0,7.

![téléchargement (1)](https://github.com/MamoudouSD/My_IA_project/assets/98142692/8918cd11-0f6f-49cc-a781-188ea55e16f2)

![téléchargement (2)](https://github.com/MamoudouSD/My_IA_project/assets/98142692/4816eccb-d6d4-4edf-acfd-e0b0face88f5)

_**Image original et image reconstruit**_
