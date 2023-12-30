# ===========================================================================
# DIANI Mamoudou Sékou
#
# Novembre 2022
#===========================================================================
# Importation des bibliothèques nécessaires pour l'implémentation de l'apprentissage par renforcement
import gym 
import classAgent 
import agent_2_Qlearning 
import agentQlearning
import argparse

# Instanciation des objets pour les agents Q-learning
dql= agent_2_Qlearning.D_Qlearning()
ql=agentQlearning.Qlearning()
a= classAgent.Agent()


# Création d'un objet ArgumentParser
# Ajout des arguments pour spécifier les paramètres de l'agent
# Analyse des arguments à partir de la ligne de commande
parser = argparse.ArgumentParser()
parser.print_help()
parser.add_argument('alpha', type=float, help='La valeur de alpha (float), le taux d\'apprentissage')
parser.add_argument('gamma', type=float, help='La valeur de gamma (float), le facteur de remise')
parser.add_argument('epsilon', type=float, help='La valeur de epsilon (float), Quantité aléatoire dans la sélection d\'action ')
parser.add_argument('epsilon_decay', type=float, help='La valeur de epsilon decay (float), Montant fixe à diminuer')
parser.add_argument('epsilon_min', type=float, help='La valeur de epsilon minimum (float)')
parser.add_argument('x', type=int, help='La valeur de x (int), Le nombre de fois que l\'agent doit jouer sur le problème')
parser.add_argument('format', type=str, help='le format, soit 4x4 ou 8x8')
parser.add_argument('type', type=str, help='le type d\'agent, soit Q_learning ou le double_Q_learning')
args=parser.parse_args()


# Vérification des types et des valeurs des arguments passés en ligne de commande
    # Si les types et les valeurs sont valides, on verifie la condition si le type est Q_learning
        # Alors modification des attributs de l'agent Q_learning par les arguments passés en ligne de commande
        # ensuite exécution de l'entraînement et affichage des résultats avec la classe  Q_learning
    # Sinon
        # Modification  des attributs de l'agent double Q_learning par les arguments passés en ligne de commande
        # Exécution de l'entraînement et affichage des résultats
# Si les types ou les valeurs ne sont pas valides, afficher un message d'erreur
if (isinstance(args.alpha, float) & isinstance(args.gamma, float) & isinstance(args.epsilon, float) & isinstance(args.epsilon_decay, float) & isinstance(args.epsilon_min, float) & isinstance(args.x, int) & (args.format=='4x4' or args.format=='8x8') & (args.type=='Q_learning' or args.type=='double_Q_learning')):
    if (args.type == 'Q_learning'):
        ql.modif_parametre(args.alpha, args.gamma, args.epsilon, args.epsilon_decay, args.epsilon_min, args.x, args.format, args.type)
        nbs= ql.play()
        print("**************************************************************")
        print ("le nombre de succes pour ", args.x," est: ",nbs)
        print("**************************************************************")
    else:
        dql.modif_parametre(args.alpha, args.gamma, args.epsilon, args.epsilon_decay, args.epsilon_min, args.x, args.format, args.type)
        nbsA, nbsB= dql.play()
        print("**************************************************************")
        print ("le nombre de succes avec qtable A pour ", args.x," est: ",nbsA)
        print ("le nombre de succes avec qtable B pour ", args.x," est: ",nbsB)
        print("**************************************************************")
else:
    print("Vérifier les valeurs entrée si ceux sont les bon types et/ou les bonnes valeurs")
    print ("Taper la commande <<python main.py -h>> pour plus d'information")
