import pandas as pd
import os
from src.paths import *
from src.datasets.scaffold_split import load_and_embed_cv_ml, load_and_embed_tfidf_ml
from src.embeddings.CountVectorize import CountVectorize
from src.embeddings.TFIDF import TFIDF
from src.modelling.support_vector_machine_classifier import svm_master_class
from src.modelling.neural_net_model import NeuralNetModel
from src.modelling.neural_net import NeuralNet
from tqdm import tqdm

NUM_SAMPLES = 1


experiment_folder = os.path.join(LOG_DIRECTORY, "experiment11")
if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)


load_and_embed_cv_ml(dataset="bace")

label_df = pd.read_csv(os.path.join(SMILES_DIRECTORY, "cleaned_bace_ml.csv"))
train_labels = label_df[label_df["set"] == "train"]["classification"].to_numpy()
test_labels = label_df[label_df["set"] == "test"]["classification"].to_numpy()
valid_labels = label_df[label_df["set"] == "valid"]["classification"].to_numpy()

embeddings = CountVectorize.load_embedding(os.path.join(EMBEDDING_DIRECTORY, "bace_ml_cv.embeddding.npy"))
train_embeddings = embeddings[label_df["set"] == "train", :]
test_embeddings = embeddings[label_df["set"] == "test", :]
valid_embeddings = embeddings[label_df["set"] == "valid", :]

metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                          index=range(NUM_SAMPLES))

for sample in tqdm(range(NUM_SAMPLES)):
    experiment = {
        "experiment": f"experiment11_sample",
        "batch_size": 200,
        "input_layer": train_embeddings.shape[1],
        "learning_rate": 0.001,
        "max_epochs": 50,
        "device": "cuda",
        "model_name": f"experiment11_bace_cv_nn",
        "shuffle_data": True,
        "hot_start_file_name": None,
        "log_metrics": False,
        "save_checkpoints": True
    }

    model = NeuralNetModel(input_layer_dim=experiment["input_layer"])
    trainer = NeuralNet(model, experiment)
    val_metrics = trainer.main_train_test_on_data(train_data=train_embeddings, test_data=valid_embeddings,
                                                   train_labels=train_labels, test_labels=valid_labels)


    models = os.listdir(os.path.join(MODEL_DIRECTORY, experiment["experiment"] + "/" + experiment["model_name"]))
    paths = [os.path.join(os.path.join(MODEL_DIRECTORY, experiment["experiment"] + "/" + experiment["model_name"]), basename) for basename in models]
    best = max(paths, key=os.path.getctime)

    test_experiment = {
        "experiment": "experiment11",
        "batch_size": 200,
        "input_layer": train_embeddings.shape[1],
        "learning_rate": 0.001,
        "max_epochs": 50,
        "device": "cuda",
        "model_name": f"experiment11_bace_cv_nn",
        "shuffle_data": True,
        "hot_start_file_name": best,
        "log_metrics": False,
        "save_checkpoints": True
    }
    test_metrics = trainer.main_test_on_data(test_data=test_embeddings, test_labels=test_labels)

    metrics_df.iloc[sample] = [test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"],
                               test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]]

test_metrics_file_name = os.path.join(experiment_folder,
                                      "experiment11_bace_cv_nn.csv")
metrics_statistics_file_name = os.path.join(experiment_folder,
                                            "experiment11_bace_cv_nn_stats.csv")
metrics_df.to_csv(test_metrics_file_name)
metrics_df = metrics_df.apply(pd.to_numeric)
statistics_df = metrics_df.describe(include='all')
statistics_df.to_csv(metrics_statistics_file_name)



# BBBP

load_and_embed_cv_ml(dataset="BBBP")

label_df = pd.read_csv(os.path.join(SMILES_DIRECTORY, "cleaned_BBBP_ml.csv"))
train_labels = label_df[label_df["set"] == "train"]["classification"].to_numpy()
test_labels = label_df[label_df["set"] == "test"]["classification"].to_numpy()
valid_labels = label_df[label_df["set"] == "valid"]["classification"].to_numpy()

embeddings = TFIDF.load_embedding(os.path.join(EMBEDDING_DIRECTORY, "BBBP_ml_cv.embeddding.npy"))
train_embeddings = embeddings[label_df["set"] == "train", :]
test_embeddings = embeddings[label_df["set"] == "test", :]
valid_embeddings = embeddings[label_df["set"] == "valid", :]

metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                          index=range(NUM_SAMPLES))

for sample in tqdm(range(NUM_SAMPLES)):
    experiment = {
        "experiment": f"experiment11_sample",
        "batch_size": 200,
        "input_layer": train_embeddings.shape[1],
        "learning_rate": 0.001,
        "max_epochs": 50,
        "device": "cuda",
        "model_name": f"experiment11_BBBP_cv_nn",
        "shuffle_data": True,
        "hot_start_file_name": None,
        "log_metrics": False,
        "save_checkpoints": True
    }

    model = NeuralNetModel(input_layer_dim=experiment["input_layer"])
    trainer = NeuralNet(model, experiment)
    val_metrics = trainer.main_train_test_on_data(train_data=train_embeddings, test_data=valid_embeddings,
                                                  train_labels=train_labels, test_labels=valid_labels)

    models = os.listdir(os.path.join(MODEL_DIRECTORY, experiment["experiment"] + "/" + experiment["model_name"]))
    paths = [os.path.join(os.path.join(MODEL_DIRECTORY, experiment["experiment"] + "/" + experiment["model_name"]),
                          basename) for basename in models]
    best = max(paths, key=os.path.getctime)

    test_experiment = {
        "experiment": "experiment11",
        "batch_size": 200,
        "input_layer": train_embeddings.shape[1],
        "learning_rate": 0.001,
        "max_epochs": 50,
        "device": "cuda",
        "model_name": f"experiment11_BBBP_cv_nn",
        "shuffle_data": True,
        "hot_start_file_name": best,
        "log_metrics": False,
        "save_checkpoints": True
    }
    test_metrics = trainer.main_test_on_data(test_data=test_embeddings, test_labels=test_labels)

    metrics_df.iloc[sample] = [test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"],
                               test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]]

test_metrics_file_name = os.path.join(experiment_folder,
                                      "experiment11_BBBP_cv_nn.csv")
metrics_statistics_file_name = os.path.join(experiment_folder,
                                            "experiment11_BBBP_cv_nn_stats.csv")
metrics_df.to_csv(test_metrics_file_name)
metrics_df = metrics_df.apply(pd.to_numeric)
statistics_df = metrics_df.describe(include='all')
statistics_df.to_csv(metrics_statistics_file_name)
