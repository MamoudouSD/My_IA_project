# ===========================================================================
# DIANI Mamoudou Sékou
#
# Novembre 2022
#===========================================================================
# Importation des bibliothèques nécessaires pour l'implémentation de l'apprentissage par renforcement
import gym
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map


class Agent:
    
    # Initialisation des attributs spécifiques
    def __init__(self):
        self.attributes = {'alpha': 0.1, 'gamma': 0.95, 'epsilon': 1.0,
                           'epsilon_decay': 0.001, 'epsilon_min': 0.05, 'x': 100, 'format': '4x4', 'type':'double_Q_learning'}
    

    # Méthode pour obtenir une représentation sous forme de chaîne
    def __str__(self):
        return "Classe mere"


    # Fonction qui affiche les attributs
    def recup_parametre(self):
        return self.attributes


    # Fonction qui modifie les attributs
    def modif_parametre(self, a, g, e, ed, em, x, f, t):
        self.attributes['alpha'] = a
        self.attributes['gamma'] = g
        self.attributes['epsilon'] = e
        self.attributes['epsilon_decay'] = ed
        self.attributes['epsilon_min'] = em
        self.attributes['x'] = x
        self.attributes['format'] = f
        self.attributes['type'] = t
        return True


    # Cette fonction initialise l'environnement et les tables Q en fonction des attributs et retourne l'environnement et la/les table
    # Elle sélection le format de la grille de l'environnement (4x4 ou 8x8) ensuite on a la
    # Création de l'instance de l'environnement FrozenLake
    # Réinitialisation de l'environnement à son état initial
    # Initialisation des tables Q en fonction du type d'algorithme
    def init_environment(self):
        if self.attributes['format'] == '4x4':
            formatFL = 'FrozenLake-v1'
        else:
            formatFL = 'FrozenLake8x8-v1'
        environment = gym.make(formatFL, desc=None, is_slippery=True)
        environment.reset()
        if self.attributes['type']== 'Q_learning':
            qtable = np.zeros((environment.observation_space.n, environment.action_space.n))
            return environment, qtable
        else:
            qtableA = np.zeros((environment.observation_space.n, environment.action_space.n))
            qtableB = np.zeros((environment.observation_space.n, environment.action_space.n))
            return environment, qtableA, qtableB


    # Cette fonction choisit une action en fonction de la politique epsilon-greedy
    # Générer un nombre aléatoire entre 0 et 1
    # Si le nombre aléatoire est inférieur à epsilon, choisir une action aléatoire
    # Sinon, choisir l'action avec la valeur la plus élevée dans l'état actuel
    # Retourner l'action choisie
    def choix_action(self, qtable, state, environment):
        rnd = np.random.random()
        if rnd < self.attributes['epsilon']:
            action = environment.action_space.sample()
        else:
            action = np.argmax(qtable[state])
        return action
