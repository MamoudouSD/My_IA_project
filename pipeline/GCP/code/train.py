# package import
#from tensorflow.python.framework import dtypes
from tensorflow_io.bigquery import BigQueryClient
import tensorflow as tf
from google.cloud import bigquery
from google.cloud import aiplatform
import argparse
import os
import numpy as np


parser = argparse.ArgumentParser()
# the passed param, dest: a name for the param, default: if absent fetch this param from the OS, type: type to convert to, help: description of argument
parser.add_argument('--epochs', dest = 'epochs', default = 10, type = int, help = 'Number of Epochs')
parser.add_argument('--batch_size', dest = 'batch_size', default = 32, type = int, help = 'Batch Size')
parser.add_argument('--var_target', dest = 'var_target', type=str)
parser.add_argument('--var_target1', dest = 'var_target1', type=str)
parser.add_argument('--project_id', dest = 'project_id', type=str)
parser.add_argument('--bq_project', dest = 'bq_project', type=str)
parser.add_argument('--bq_dataset', dest = 'bq_dataset', type=str)
parser.add_argument('--bq_table', dest = 'bq_table', type=str)
parser.add_argument('--bq_table_train', dest = 'bq_table_train', type=str)
parser.add_argument('--bq_table_test', dest = 'bq_table_test', type=str)
parser.add_argument('--region', dest = 'region', type=str)
parser.add_argument('--experiment', dest = 'experiment', type=str)
parser.add_argument('--series', dest = 'series', type=str)
parser.add_argument('--experiment_name', dest = 'experiment_name', type=str)
parser.add_argument('--run_name', dest = 'run_name', type=str)
args = parser.parse_args()

# clients
bq = bigquery.Client(project = args.project_id)
aiplatform.init(project = args.project_id, location = args.region)


###on verifie si l'essaie args.run_name existe dans args.experiment_name
####s'il existe, on cree une instance
####sinon, on crée un nouvel essai d'expérience
###on enregistre des paramètres associés à cet essai d'expérience. 7 
if args.run_name in [run.name for run in aiplatform.ExperimentRun.list(experiment = args.experiment_name)]:
    expRun = aiplatform.ExperimentRun(run_name = args.run_name, experiment = args.experiment_name)
else:
    expRun = aiplatform.ExperimentRun.create(run_name = args.run_name, experiment = args.experiment_name)

expRun.log_params({'experiment': args.experiment, 'series': args.series, 'project_id': args.project_id})


###recuperation de donnees de train


query = f"""SELECT * FROM `{args.bq_project}.{args.bq_dataset}.{args.bq_table_train}`"""
a=bq.query(query = query).to_dataframe()

x = a.iloc[:, :-1]

x = np.array(x)
X_train = np.reshape(x, (x.shape[0], x.shape[1], 1))

y = a.iloc[:,-1]
Y_train = np.array(y)

expRun.log_params({'data_source_training': f'bq://{args.bq_project}.{args.bq_dataset}.{args.bq_table_train}', 'type': 'prediction', 'var_target': args.var_target})

###recuperation de donnees de test

query = f"""SELECT * FROM `{args.bq_project}.{args.bq_dataset}.{args.bq_table_test}`"""
a=bq.query(query = query).to_dataframe()

x = a.iloc[:, :-1]
x = np.array(x)
X_test = np.reshape(x, (x.shape[0], x.shape[1], 1))

y = a.iloc[:,-1]
Y_test = np.array(y)

expRun.log_params({'data_source_testing': f'bq://{args.bq_project}.{args.bq_dataset}.{args.bq_table_test}', 'type': 'prediction', 'var_target': args.var_target1})


#definition du model


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units = 80, activation = 'relu'))

model.add(tf.keras.layers.Dense(units = 1, activation = 'tanh'))

model.compile(
    optimizer = tf.keras.optimizers.RMSprop(), #SGD or Adam
    loss = tf.keras.losses.BinaryCrossentropy()
)


# Compilation et training du modèle
model.compile(optimizer='RMSprop', loss = 'binary_crossentropy')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR'], histogram_freq=1)
history = model.fit(X_train, Y_train, epochs = args.epochs, callbacks = [tensorboard_callback])
expRun.log_params({'training.epochs': history.params['epochs']})

for e in range(0, history.params['epochs']):
    expRun.log_time_series_metrics(
        {
            'train_loss': history.history['loss'][e]
        }
    )

#####Evaluation du model

loss = model.evaluate(X_test, Y_test)
expRun.log_metrics({'test_loss': loss})

# training evaluations:
loss = model.evaluate(X_train, Y_train)
expRun.log_metrics({'train_loss': loss})

#####Sauvegarde du model

model.save(os.getenv("AIP_MODEL_DIR"))
expRun.log_params({'model.save': os.getenv("AIP_MODEL_DIR")})
expRun.end_run()