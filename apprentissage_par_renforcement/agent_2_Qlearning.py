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
from classAgent import Agent

class D_Qlearning(Agent):

    # Recuperation des attributs du parent
    def __init__(self):
        super().__init__()
    

    # Cette fonction définit le choix d'action dans le cadre de l'algorithme Double Q-learning.
    # La fonction prend en parametre les tables Q_1 et Q_2, l'etat state, l'environnement et
    # i un entier entre 0 et le nombre de fois que dois jouer l'agent
    # Elle génère un nombre aléatoire entre 0 et 1
    # Si le nombre aléatoire est inférieur à l'attribut epsilon, on choisit une action aléatoire
    # Sinon, on choisit l'action avec la plus grande valeur dans l'état actuel et on utilise
    # la formule Double Q-learning pour calculer la valeur de chaque action
    def choix_action(self, Q_1, Q_2, state, env, i):
        rnd = np.random.random()
        if rnd < self.attributes['epsilon']:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_1[state, :] + Q_2[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        return action
        

    # Cette fonction met à jour les tables dans le cadre de l'algorithme Double Q-learning
    # La fonction prend en parametre l'action, les tables qtableA et qtableB, l'environnement, l'etat state
    # Elle effectue l'action dans l'environnement et obtient les nouvelles informations
    # Elle sélectionne aléatoirement quelle table Q mettre à jour, la table Q_A ou La table Q_B 
    # met à jour l'état actuel et retourne la récompense, l'état d'achèvement et le nouvel état
    def maj_qtable(self, action, qtableA, qtableB, environment, state):
        new_state, reward, done, _, _ = environment.step(action)
        if np.random.random()>0.5:
            qtableA[state, action] = qtableA[state, action] + self.attributes['alpha'] * \
                (reward + self.attributes['gamma'] *
                qtableB[new_state, np.argmax(qtableA[new_state, :])] - qtableA[state, action])
        else:
            qtableB[state, action] = qtableB[state, action] + self.attributes['alpha'] * \
                (reward + self.attributes['gamma'] *
                qtableA[new_state, np.argmax(qtableB[new_state, :])] - qtableB[state, action])
        state = new_state
        return reward, done,state


    # Fonction d'entraînement pour l'algorithme Double Q-learning qui retourne l'environment, qtableA, qtableB
    # Elle initialise l'environnement de jeu et les tables Q_A et Q_B et 
    # crée une liste pour enregistrer les résultats de chaque épisode (Succès)
    # épisode est un entier qui varie de 0 au nombre de fois de jeu
    # Une boucle d'entraînement est effectuée sur un nombre spécifié d'épisodes (attribut x)
        # durant la quelle on réinitialise de l'état de l'environnement,
        # initialise du résultat de l'épisode à "Échec"
        # Une Boucle d'interaction est lancée avec l'environnement jusqu'à la fin de l'épisode
            # On choisit l'action en utilisant la fonction choix_action
            # On met à jour de la table Q en fonction de l'action choisie grace a la fonction maj_qtable
            # Si une récompense est obtenue, on marque l'épisode comme "Succès"
            # On met à jour de l'exploration epsilon (la Quantité de la sélection d'action)
    # Retourner l'environnement final et les tables Q_A et Q_B
    def entrainement(self):
        environment, qtableA, qtableB= self.init_environment()
        outcomes = []
        for i in range(self.attributes['x']):
            state = environment.reset()
            done = False
            state = state[0]
            outcomes.append("Failure")
            while not done:
                action = self.choix_action(qtableA, qtableB, state, environment, i)
                reward, done, state = self.maj_qtable(action, qtableA, qtableB, environment, state)
                if reward:
                    outcomes[-1] = "Success"
            self.attributes['epsilon'] = max(
                self.attributes['epsilon'] - self.attributes['epsilon_decay'], self.attributes['epsilon_min'])
        return environment, qtableA, qtableB


    # Cette fonction évalue les performances de l'agent après l'entraînement et renvoie le nombre de succes
    # Elle entraîne l'agent et obtient l'environnement final et les tables Q_A, Q_B
    # On initialise les compteurs de succès pour Q_A et Q_B
    # On évalue la politique basée sur la table Q_A sur 100 épisodes
            # on compte le nombre succes (reward = 1 si c'est le but et 0 sinon)
    # On évalue la politique basée sur la table Q_B sur 100 épisodes
            # on compte le nombre succes
    # Retourner le nombre total de succès pour Q_A et Q_B
    def play(self):
        environment, qtableA, qtableB=self.entrainement()
        nb_succesA=0
        nb_succesB=0
        for i in range(100):
            state=environment.reset()
            done = False
            state = state[0]
            while not done:
                action=np.argmax(qtableA[state])
                new_state, reward, _, _, _ = environment.step(action)
                state=new_state
                nb_succesA += reward
        for i in range(100):
            state=environment.reset()
            done = False
            state = state[0]
            while not done:
                action=np.argmax(qtableB[state])
                new_state, reward, _, _, _ = environment.step(action)
                state=new_state
                nb_succesB += reward
        return nb_succesA, nb_succesB
        