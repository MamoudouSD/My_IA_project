# Introduction 

Le but de ce TP est de réaliser une IA appelé agent qui a pour objectif de résoudre
l'environnement lac gelé en utilisant l'apprentissage par renforcement. Cet agent doit prendre
des décisions qui vont le mener d’un point initial au terminus en minimisant le nombre d’action

# Principe du jeu

#### Lac gelé 4x4
Dans la configuration 4x4 du jeu FrozenLake, comprenant 16 tuiles, notre agent peut occuper l'une des 16 positions différentes, désignées comme états. Chaque état offre 4 actions possibles : aller à gauche, en bas, à droite et en haut. L'objectif de notre agent est d'apprendre quelle action choisir dans chaque état. Pour déterminer quelle action est optimale dans un état donné, nous attribuons une valeur de qualité à chaque action. Avec 16 états et 4 actions, nous avons un total de 16 * 4 = 64 valeurs à calculer.

La représentation appropriée de ces valeurs consiste à utiliser une table appelée Q-table, où les lignes représentent les différents états 's' et les colonnes représentent les différentes actions 'a'. Chaque cellule de cette Q-table contient une valeur Q(s, a), représentant la qualité de l'action 'a' dans l'état 's' (1 si c'est la meilleure action possible, 0 si c'est une action inefficace).
#### Lac gelé 8x8
Il s’agit du même principe que le lac gelé 4x4, sauf que celui du 8x8 possède 64 tuiles donc nous devons calculer 64*4=256 valeurs.


# Structure du code

Le code est structuré en trois classes. La première, appelée "ClassAgent", est une classe mère comprenant des attributs représentant les hyperparamètres, avec des valeurs par défaut stockées dans un dictionnaire. Les principales méthodes de cette classe incluent "Recup_parametre()", qui renvoie le dictionnaire d'attributs, "Modif_parametre()" pour modifier les attributs, et "Init_environment()" pour initialiser l'environnement en utilisant la bibliothèque gym.

Les deux classes enfants correspondent à deux types d'agents : l'agent Q-learning et l'agent double Q-learning. L'agent Q-learning possède des méthodes telles que "Maj_qtable()" pour mettre à jour la table Q, "Entrainement()" pour l'entraînement et "Play()" pour jouer le jeu et évaluer les succès. L'agent double Q-learning hérite des méthodes de la classe mère et possède ses propres implémentations, notamment pour le choix d'action et la mise à jour des tables.

L'agent Q-learning utilise une approche classique de mise à jour de la table Q, tandis que l'agent double Q-learning met à jour deux tables avec une probabilité de 50% chacune. L'algorithme d'apprentissage double Q utilise deux estimations indépendantes pour maximiser l'efficacité de l'apprentissage. La méthode "Entrainement()" des deux agents effectue des entraînements multiples, prend des décisions basées sur le choix d'action et met à jour les tables jusqu'à ce que les conditions de fin soient atteintes.

# Résultat

## Agent Q_learning

#### Lac gelé 4x4

Hyper paramètre par défaut (code argparse : python main.py 0.1 0.95 1.0 0.95 0.05 1000 '4x4' 'Q_learning' ) :
  - x = 1000
  - Alpha = 0.1
  - Gamma = 0.95
  - Epsilon = 1.0
  - Epsilon_decay = 0.95
  - Epsilon_min = 0.05

En utilisant les paramètre par défaut, le nombre de succès obtenu est de 0%. Cela est dû à la baisse rapide de la valeur d’epsilon.

Hyper paramètre que nous avons choisis (code argparse : python main.py 0.1 0.95 1.0 0.001 0.05 1000 '4x4' 'Q_learning' ) :
  - x = 1000
  - Alpha = 0.1
  - Gamma = 0.95
  - Epsilon = 1.0
  - Epsilon_decay = 0.001
  - Epsilon_min = 0.05

En utilisant les paramètre que nous avons choisi, le nombre de succès obtenu est de 63%. Juste une modification d’un hyper paramètre à permis d’améliorer le résultat.

#### Lac gelé 8x8
Hyper paramètre par défaut (code argparse : python main.py 0.1 0.95 1.0 0.95 0.05 1000 '8x8' 'Q_learning' ) :
  - X = 1000
  - Alpha = 0.1
  - Gamma = 0.95
  - Epsilon = 1.0
  - Epsilon_decay = 0.95
  - Epsilon_min = 0.05
  - Résultat après entrainement
    
Succès sur 100=0

## Agent double_Q_learning

#### Lac gelé 4x4

Hyper paramètre par défaut (code argparse : python main.py 0.1 0.95 1.0 0.95 0.05 1000 '4x4' 'double_Q_learning' ) :
  - x = 1000
  - Alpha = 0.1
  - Gamma = 0.95
  - Epsilon = 1.0
  - Epsilon_decay = 0.95
  - Epsilon_min = 0.05
    
Le nombre de succès avec qtable A pour 100 est: 9.0%
Le nombre de suces avec qtable B pour 100 est : 5.0%

Hyper paramètre que nous avons choisis (code argparse : python main.py 0.1 0.95 1.0 0.001 0.05 1000 '4x4' 'double_Q_learning' ) :
  - x = 1000
  - Alpha = 0.1
  - Gamma = 0.95
  - Epsilon = 1.
  - Epsilon_decay = 0.001
  - Epsilon_min = 0.05

Le nombre de succès avec qtable A pour 100 est : 13.0%
Le nombre de succès avec qtable B pour 100 est : 4.0%



