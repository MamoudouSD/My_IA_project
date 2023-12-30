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

class Qlearning(Agent):

    # Recuperation des attributs du parent
    def __init__(self):
        super().__init__()
    
    
    # Cette fonction met à jour la table dans le cadre de l'algorithme 
    # La fonction prend en parametre l'action, la table qtable, l'environnement, l'etat state
    # Elle effectue l'action dans l'environnement et obtient les nouvelles informations
    # Elle met à jour la table, l'état actuel et retourne la récompense, l'état d'achèvement et le nouvel état
    def maj_qtable(self, action, qtable, environment, state):
        new_state, reward, done, _, _ = environment.step(action)
        qtable[state, action] = qtable[state, action] + self.attributes['alpha'] * \
            (reward + self.attributes['gamma'] *
             np.max(qtable[new_state]) - qtable[state, action])
        state = new_state
        return reward, done,state


    # Fonction d'entraînement pour l'algorithme Q-learning qui retourne l'environment, qtable
    # Elle initialise l'environnement de jeu et la table qtable 
    # crée une liste pour enregistrer les résultats de chaque épisode (Succès)
    # épisode est un entier qui varie de 0 au nombre de fois de jeu
    # Une boucle d'entraînement est effectuée sur un nombre spécifié d'épisodes (attribut x)
        # durant la quelle on réinitialise de l'état de l'environnement,
        # initialise du résultat de l'épisode à "Échec"
        # Une Boucle d'interaction est lancée avec l'environnement jusqu'à la fin de l'épisode
            # On choisit l'action en utilisant la fonction choix_action
            # On met à jour de la table en fonction de l'action choisie grace a la fonction maj_qtable
            # Si une récompense est obtenue, on marque l'épisode comme "Succès"
            # On met à jour de l'exploration epsilon (la Quantité de la sélection d'action)
    # Retourner l'environnement final et la table qtable
    def entrainement(self):
        environment, qtable= self.init_environment()
        outcomes = []
        for i in range(1000):
            state = environment.reset()
            done = False
            state = state[0]
            outcomes.append("Failure")
            while not done:
                action = self.choix_action(qtable, state, environment)
                reward, done, state = self.maj_qtable(action, qtable, environment, state)
                if reward:
                    outcomes[-1] = "Success"
            self.attributes['epsilon'] = max(
                self.attributes['epsilon'] - self.attributes['epsilon_decay'], 0)
        return environment, qtable


    # Cette fonction évalue les performances de l'agent après l'entraînement et renvoie le nombre de succes
    # Elle entraîne l'agent et obtient l'environnement final et la table qtable
    # On initialise le compteur de succès 
    # On évalue la politique basée sur la table sur 100 épisodes
            # on compte le nombre succes (reward = 1 si c'est le but et 0 sinon)
    # Retourner le nombre total de succès 
    def play(self):
        environment, qtable=self.entrainement()
        nb_succes=0
        for i in range(self.attributes['x']):
            state=environment.reset()
            done = False
            state = state[0]

            while not done:
                action=np.argmax(qtable[state])
                new_state, reward, done, _, _ = environment.step(action)
                state=new_state
                nb_succes += reward
        return nb_succes
        