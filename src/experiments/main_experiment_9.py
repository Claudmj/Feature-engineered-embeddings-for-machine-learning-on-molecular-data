from src.embeddings.CountVectorize import CountVectorize
from src.embeddings.TFIDF import TFIDF
from src.embeddings.word2vec import word2vec
from src.modelling.neural_net import NeuralNet
from src.modelling.neural_net_model import NeuralNetModel
from src.paths import *
import os
import pandas as pd
from tqdm import tqdm

# Initialize what processes should be run and initialize variables
DATA_LIST = ["cleaned_binary.csv", "cleaned_bace.csv", "cleaned_BBBP.csv"]
DATA_PATHS = [FASTA_DIRECTORY, SMILES_DIRECTORY, SMILES_DIRECTORY]
DATA_TYPES = ["FASTA", "SMILES", "SMILES"]
DATASET_NAMES = ["FASTA", "bace", "BBBP"]
NUM_SAMPLES = 50


experiment_folder = os.path.join(LOG_DIRECTORY, "experiment9")
if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

for i in tqdm(range(len(DATA_LIST))):
    df = pd.read_csv(os.path.join(DATA_PATHS[i], DATA_LIST[i]))
    labels = df["classification"].to_numpy()

    if DATA_TYPES[i] == "FASTA":
        doc_list = df["sequence"].to_list()
    else:
        doc_list = df["sentence"].to_list()


    metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                              index=range(NUM_SAMPLES))
    for sample in tqdm(range(NUM_SAMPLES)):
        # CV
        cv = CountVectorize(doc_list=doc_list, data_type=DATA_TYPES[i])

        embeddings = cv.embedding

        experiment = {
            "experiment": "experiment9",
            "batch_size": 200,
            "input_layer": embeddings.shape[1],
            "learning_rate": 0.001,
            "max_epochs": 50,
            "device": "cuda",
            "model_name": "experiment9_" + DATASET_NAMES[i] + "_cv_nn",
            "shuffle_data": True,
            "hot_start_file_name": None,
            "log_metrics": False,
            "save_checkpoints": False,
            "data": embeddings,
            "labels": labels
        }

        model = NeuralNetModel(input_layer_dim=experiment["input_layer"])
        trainer = NeuralNet(model, experiment)
        test_metrics = trainer.main_train_test()
        metrics_df.iloc[sample] = [test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]]

    test_metrics_file_name = os.path.join(experiment_folder, "experiment9_" + DATASET_NAMES[i] + "_cv_nn.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder, "experiment9_" + DATASET_NAMES[i] + "_cv_nn_stats.csv")
    metrics_df.to_csv(test_metrics_file_name)
    metrics_df = metrics_df.apply(pd.to_numeric)
    statistics_df = metrics_df.describe(include='all')
    statistics_df.to_csv(metrics_statistics_file_name)


    metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                              index=range(NUM_SAMPLES))
    for sample in tqdm(range(NUM_SAMPLES)):
        tfidf = TFIDF(doc_list=doc_list, data_type=DATA_TYPES[i])

        embeddings = tfidf.embedding

        experiment = {
            "experiment": "experiment9",
            "batch_size": 200,
            "input_layer": embeddings.shape[1],
            "learning_rate": 0.001,
            "max_epochs": 50,
            "device": "cuda",
            "model_name": "experiment9_" + DATASET_NAMES[i] + "_tfidf_nn",
            "shuffle_data": True,
            "hot_start_file_name": None,
            "log_metrics": False,
            "save_checkpoints": False,
            "data": embeddings,
            "labels": labels
        }

        model = NeuralNetModel(input_layer_dim=experiment["input_layer"])
        trainer = NeuralNet(model, experiment)
        test_metrics = trainer.main_train_test()
        metrics_df.iloc[sample] = [test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]]

    test_metrics_file_name = os.path.join(experiment_folder, "experiment9_" + DATASET_NAMES[i] + "_tfidf_nn.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder, "experiment9_" + DATASET_NAMES[i] + "_tfidf_nn_stats.csv")
    metrics_df.to_csv(test_metrics_file_name)
    metrics_df = metrics_df.apply(pd.to_numeric)
    statistics_df = metrics_df.describe(include='all')
    statistics_df.to_csv(metrics_statistics_file_name)


    # word2vec for SMILES datasets
    if DATA_TYPES[i] != "FASTA":
        metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                                  index=range(NUM_SAMPLES))
        for sample in tqdm(range(NUM_SAMPLES)):
            word2vec_model = word2vec(doc_list=doc_list, data_type=DATA_TYPES[i])

            embeddings = word2vec_model.embedding

            experiment = {
                "experiment": "experiment9",
                "batch_size": 200,
                "input_layer": embeddings.shape[1],
                "learning_rate": 0.001,
                "max_epochs": 50,
                "device": "cuda",
                "model_name": "experiment9_" + DATASET_NAMES[i] + "_word2vec_nn",
                "shuffle_data": True,
                "hot_start_file_name": None,
                "log_metrics": False,
                "save_checkpoints": False,
                "data": embeddings,
                "labels": labels
            }

            model = NeuralNetModel(input_layer_dim=experiment["input_layer"])
            trainer = NeuralNet(model, experiment)
            test_metrics = trainer.main_train_test()
            metrics_df.iloc[sample] = [test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"],
                                       test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"],
                                       test_metrics["MCC"]]

        test_metrics_file_name = os.path.join(experiment_folder, "experiment9_" + DATASET_NAMES[i] + "_word2vec_nn.csv")
        metrics_statistics_file_name = os.path.join(experiment_folder,
                                                    "experiment9_" + DATASET_NAMES[i] + "_word2vec_nn_stats.csv")
        metrics_df.to_csv(test_metrics_file_name)
        metrics_df = metrics_df.apply(pd.to_numeric)
        statistics_df = metrics_df.describe(include='all')
        statistics_df.to_csv(metrics_statistics_file_name)

print(0)