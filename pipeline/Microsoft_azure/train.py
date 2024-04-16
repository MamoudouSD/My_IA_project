# Script pour construire et entraîner un modèle de classification d'images avec Keras et TensorFlow dans Azure ML :
# 1. Installer les dépendances nécessaires via pip pour TensorFlow, joblib et Azure ML Core.
# 2. Définir les chemins d'accès aux données d'entraînement et au modèle via des arguments de ligne de commande.
# 3. Charger les données d'entraînement et de validation à partir des fichiers spécifiés.
# 4. Construire un modèle séquentiel avec des couches de convolution, d'activation, de pooling et de densité.
# 5. Configurer le modèle pour l'entraînement avec une perte MSE et l'optimiseur Adam.
# 6. Entraîner le modèle en utilisant des callbacks pour sauvegarder le meilleur modèle en fonction de la précision de validation.
# 7. Enregistrer le chemin du modèle dans Azure ML pour le suivi.

import subprocess
import argparse
import os
from azureml.core import Run
import joblib
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Input, Activation, Flatten, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint

subprocess.check_call(['pip', 'install', 'joblib'])
subprocess.check_call(['pip', 'install', 'scipy'])
subprocess.check_call(['pip', 'install', 'tensorflow'])
subprocess.check_call(['pip', 'install', 'azureml-core'])

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, dest='data_path', help='Chemin vers les données d\'entraînement')
parser.add_argument('--model-path', type=str, dest='model_path', help='Chemin vers le modele enregistrer')
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
os.makedirs(model_path, exist_ok=True)

save_model_path = os.path.join(model_path, 'best_model.hdf5')
x_train = joblib.load(os.path.join(data_path, 'x_train.pkl'))
y_train = joblib.load(os.path.join(data_path, 'y_train.pkl'))
x_val = joblib.load(os.path.join(data_path, 'x_val.pkl'))
y_val = joblib.load(os.path.join(data_path, 'y_val.pkl'))


image_scale = 100
image_channels = 3  
images_color_mode = "rgb"
image_shape = (image_scale, image_scale, image_channels)
input_layer = Input(shape=image_shape)

def feature_extraction(input):    
    x = Conv2D(64, (3, 3), padding='same')(input) 
    x = Activation("relu")(x)
    encoded = MaxPooling2D((2, 2), padding='same',strides=(2,2))(x)
    return encoded

def fully_connected(encoded):
    x = Flatten(input_shape=image_shape)(encoded)
    x = Dense(6)(x)
    sortie = Activation('softmax')(x)
    return sortie

model = Model(input_layer, fully_connected(feature_extraction(input_layer)))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
modelcheckpoint = ModelCheckpoint(filepath=save_model_path,
                                  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
classifier = model.fit(x_train, y_train,
                       epochs=10, 
                       batch_size=180, 
                       validation_data=(x_val, y_val), 
                       verbose=1, 
                       callbacks=[modelcheckpoint], 
                       shuffle=True)

run = Run.get_context()
run.log('model_path', save_model_path)