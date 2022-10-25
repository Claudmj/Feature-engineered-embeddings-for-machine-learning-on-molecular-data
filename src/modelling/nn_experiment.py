"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     :
"""

from src.modelling.neural_net import NeuralNet
from src.modelling.neural_net_model import NeuralNetModel
import pandas as pd
import os
import numpy as np
from src.paths import *

DEVICE_CPU = "cpu"
DEVICE_GPU = "cuda"
batch_size = 200
INPUT_LAYER = 10
EPOCHS = 100

df = pd.read_csv(os.path.join(FASTA_DIRECTORY, "cleaned_binary.csv"))

doc_list = df["sequence"].to_list()

labels = df["classification"].to_numpy()

embedding_file = "/home/claudio/PycharmProjects/research/data/fasta/embeddings/experiment1_FASTA_100_lda.embeddding.npy"
with open(embedding_file, "rb") as f:
    # Version
    version = np.load(f)
    embeddings = np.load(f)
experiment = {
    "experiment": "batch_size_experiment_architecture8",
    "batch_size": 200,
    "input_layer": 100,
    "hidden_layer": 5,
    "learning_rate": 0.001,
    "max_epochs": 20,
    "device": "cuda",
    "model_name": f"architecture_test",
    "shuffle_data": True,
    "hot_start_file_name": None,#os.path.join(MODEL_DIRECTORY, "experiment3", "architecture_test", "architecture_test_6.pt.tar"),
    "data": embeddings,
    "labels": labels
    }


model = NeuralNetModel(input_layer_dim=experiment["input_layer"])
trainer = NeuralNet(model, experiment)
trainer.main_train_test()