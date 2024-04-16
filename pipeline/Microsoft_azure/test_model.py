# Script pour tester un modèle de classification d'images avec TensorFlow et Azure ML :
# 1. Installer les dépendances nécessaires via pip, y compris TensorFlow, Pillow, joblib, et les SDKs Azure.
# 2. Définir les arguments de ligne de commande pour le chemin de sortie des données et du modèle.
# 3. Télécharger les images de test depuis Azure Blob Storage dans un répertoire local spécifié.
# 4. Générer un lot de données de test à partir des images téléchargées, les redimensionner et les normaliser.
# 5. Charger le modèle Keras pré-entraîné depuis le chemin spécifié.
# 6. Évaluer le modèle sur le lot de données de test et afficher les résultats d'évaluation (perte et précision).
# 7. Sauvegarder les résultats de l'évaluation dans un fichier et logger les résultats dans Azure ML pour le suivi.

import subprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import Model
import os
import argparse
import joblib
from azureml.core import Run
from azure.storage.blob import BlobServiceClient

subprocess.check_call(['pip', 'install', 'joblib'])
subprocess.check_call(['pip', 'install', 'pillow'])
subprocess.check_call(['pip', 'install', 'scipy'])
subprocess.check_call(['pip', 'install', 'tensorflow'])
subprocess.check_call(['pip', 'install', 'azureml-core'])
subprocess.check_call(['pip', 'install', 'azure-storage-blob'])


parser = argparse.ArgumentParser()
parser.add_argument('--output-data', type=str, dest='output_data', help='Chemin vers le répertoire de sortie des résultats du test')
parser.add_argument('--model-dir', type=str, dest='model_output', help='Chemin vers le répertoire de sortie du model')
args = parser.parse_args()
output_data_path = args.output_data
model_dir=args.model_output

model_path = os.path.join(model_dir, 'best_model.hdf5')

def download_blob_to_directory(account_name, account_key, container_name, local_directory_path):

    blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs()
    for blob in blobs:
        blob_name = blob.name

        if "test" in blob_name:
            relative_path = os.path.relpath(blob_name, "test")
            local_file_path = os.path.join(local_directory_path, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob_client = container_client.get_blob_client(blob_name)
            with open(local_file_path, "wb") as file:
                file.write(blob_client.download_blob().readall())

            print(f"Fichier {blob_name} téléchargé avec succès dans {local_file_path}.")

account_name = "**************"
account_key = "*************"
container_name = "***********"
local_directory_path = "test/"

download_blob_to_directory(account_name, account_key, container_name, local_directory_path)

path_data = "test"
image_scale = 100
image_channels = 3  
images_color_mode = "rgb"

test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_generator = test_data_generator.flow_from_directory(
    path_data,
    target_size=(image_scale, image_scale),
    class_mode="categorical",
    shuffle=False,
    batch_size=6000,
    color_mode=images_color_mode)


(x_test, y_test) = test_generator.next()
output_data_path="outputs"
max_value = float(x_test.max())
x_test = x_test.astype('float32') / max_value

Classifier : Model = load_model(model_path)

evaluation_result = Classifier.evaluate(x_test, y_test)

print("Perte (Loss):", evaluation_result[0])
print("Précision (Accuracy):", evaluation_result[1])

joblib.dump("Résultats du test", os.path.join(output_data_path, 'test_results.pkl'))

run = Run.get_context()
run.log('test_loss', evaluation_result[0])
run.log('test_accuracy', evaluation_result[1])