# Script pour préparer et traiter des images pour l'entraînement de modèles de machine learning avec TensorFlow et Azure ML :
# 1. Installer les packages nécessaires via pip, y compris les bibliothèques pour le traitement d'images et le stockage Azure.
# 2. Définir les paramètres d'entrée pour le chemin des données de sortie via des arguments de ligne de commande.
# 3. Définir une fonction pour télécharger des images spécifiques depuis Azure Blob Storage vers un répertoire local.
# 4. Préparer les générateurs de données pour l'entraînement et la validation avec augmentation des données.
# 5. Charger et normaliser un lot d'images d'entraînement et de validation.
# 6. Sauvegarder les données d'entraînement et de validation dans des fichiers locaux.
# 7. Logger les chemins des fichiers de données dans Azure ML pour suivi et référence.

import os
import argparse
import subprocess
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azureml.core import Run


subprocess.check_call(['pip', 'install', 'joblib'])
subprocess.check_call(['pip', 'install', 'pillow'])
subprocess.check_call(['pip', 'install', 'scipy'])
subprocess.check_call(['pip', 'install', 'tensorflow'])
subprocess.check_call(['pip', 'install', 'azureml-core'])
subprocess.check_call(['pip', 'install', 'azure-storage-blob'])


parser = argparse.ArgumentParser()
parser.add_argument('--output-data', type=str, dest='output_data', help='Chemin vers le dossier de sortie')
args = parser.parse_args()
output_data_path = args.output_data


def download_blob_to_directory(account_name, account_key, container_name, local_directory_path):

    blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs()
    for blob in blobs:
        blob_name = blob.name

        if "entrainement" in blob_name:
            relative_path = os.path.relpath(blob_name, "entrainement")
            local_file_path = os.path.join(local_directory_path, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob_client = container_client.get_blob_client(blob_name)
            with open(local_file_path, "wb") as file:
                file.write(blob_client.download_blob().readall())

            print(f"Fichier {blob_name} téléchargé avec succès dans {local_file_path}.")

account_name = "************"
account_key = "**********"
container_name = "********"
local_directory_path = "entrainement/"

download_blob_to_directory(account_name, account_key, container_name, local_directory_path)

path_data = "entrainement"
image_scale = 100
image_channels = 3  
images_color_mode = "rgb"

training_data_generator = ImageDataGenerator( rescale=1. / 255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, validation_split=0.25)



training_generator = training_data_generator.flow_from_directory(
    path_data,
    color_mode=images_color_mode, 
    target_size=(image_scale, image_scale),
    batch_size= 18000  , 
    class_mode="categorical",
    shuffle=True, 
    subset='training')

validation_generator = training_data_generator.flow_from_directory(
    path_data, 
    color_mode=images_color_mode, 
    target_size=(image_scale, image_scale),
    batch_size=6000,
    class_mode="categorical",
    shuffle=True, 
    subset='validation')

(x_train, y_train) = training_generator.next()
(x_val, y_val) = validation_generator.next()

max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_val = x_val.astype('float32') / max_value

os.makedirs(output_data_path, exist_ok=True)

output_file_x_train = os.path.join(output_data_path, 'x_train.pkl')
output_file_y_train = os.path.join(output_data_path, 'y_train.pkl')
output_file_x_val = os.path.join(output_data_path, 'x_val.pkl')
output_file_y_val = os.path.join(output_data_path, 'y_val.pkl')

joblib.dump(x_train, output_file_x_train)
joblib.dump(y_train, output_file_y_train)
joblib.dump(x_val, output_file_x_val)
joblib.dump(y_val, output_file_y_val)


run = Run.get_context()
run.log('output_data_x_train', output_file_x_train)
run.log('output_data_y_train', output_file_y_train)
run.log('output_data_x_val', output_file_x_val)
run.log('output_data_y_val', output_file_y_val)