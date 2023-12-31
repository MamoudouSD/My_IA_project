# Introduction
L'objectif principal de cette étude était de développer un réseau de neurones convolutif (CNN) capable de classifier des images appartenant à six espèces animales différentes. Plus précisément, l'étude visait à comprendre et à manipuler divers paramètres pour observer leurs impacts sur les performances du modèle CNN. De plus, un objectif clé était d'atteindre un niveau élevé de précision dans la classification des images des espèces animales spécifiées, qui comprennent l'Éléphant, la Girafe, le Léopard, le Rhinocéros, le Tigre et le Zèbre.

#Architecture du Modèle

Le modèle CNN construit comprend deux parties principales : la partie d'extraction de caractéristiques et la partie entièrement connectée.

#### Partie d'extraction de caractéristiques :

2 couches Conv2D avec 100 filtres, un noyau de 3x3, et une activation ReLU.
Maxpooling2D avec un pas de 2x2 et un noyau de 2x2.
Normalisation par lots.
3 couches Conv2D avec 128 filtres, un noyau de 3x3, et une activation ReLU.
Maxpooling2D avec un pas de 2x2 et un noyau de 2x2.
Normalisation par lots.
4 couches Conv2D avec 256 filtres, un noyau de 3x3, et une activation ReLU.
Maxpooling2D avec un pas de 2x2 et un noyau de 2x2.
Normalisation par lots.
4 couches Conv2D avec 256 filtres, un noyau de 3x3, et une activation ReLU.
Maxpooling2D avec un pas de 2x2 et un noyau de 2x2.
Normalisation par lots.
4 couches Conv2D avec 512 filtres, un noyau de 3x3, et une activation ReLU.
Maxpooling2D avec un pas de 2x2 et un noyau de 2x2.
Normalisation par lots.

#### Partie entièrement connectée :

Aplatir (Flatten).
Couche Dense avec 512 unités et activation ReLU.
Abandon (Dropout) de 0,5.
Couche Dense avec 512 unités et activation ReLU.
Abandon (Dropout) de 0,5.
Couche Dense avec 6 unités (nombre de classes) et activation softmax.

# Optimisation du Modèle
Partie, d'une seule couche de convolution et une seule couche dense, nous avons observé que les performances du modèle s'amélioraient à mesure que le nombre total de paramètres augmentait. Donc nous décidé d'augmenter le nombre de couches de convolution, dense et de filtres. Face a une stagnation de l'évolution, l'approche a évolué vers l'augmentation de la taille des images. Cette décision a été influencée par l'observation selon laquelle des CNN bien connus tels que VGG19 et AlexNet utilisent des images de plus grande taille. La taille de l'image a été augmentée progressivement, avec la limitation étant les contraintes de RAM de Google Colab. Malgré cette limitation, nous avons atteint la précision de 89%.

_![téléchargement (1)](https://github.com/MamoudouSD/My_IA_project/assets/98142692/658c47f0-4b3d-4df4-8953-e2328f9b284d)_
_**Courbe de precision**_

_![téléchargement (2)](https://github.com/MamoudouSD/My_IA_project/assets/98142692/32efbe84-f50d-4875-9c70-fac6a701e607)_
_**Courbes de perte**_
