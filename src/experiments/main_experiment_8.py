"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 14 September 2022
@Version  : 0.1
@Desc     : This file will be used to test classes and functions in development phase.
"""

from src.embeddings.CountVectorize import CountVectorize
from src.embeddings.TFIDF import TFIDF
from src.embeddings.word2vec import word2vec
from src.modelling.naive_bayes_classifier import nbc_master_class
from sklearn.model_selection import train_test_split
from src.paths import *
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


# Initialize what processes should be run and initialize variables
DATA_LIST = ["cleaned_binary.csv", "cleaned_bace.csv", "cleaned_BBBP.csv"]
DATA_PATHS = [FASTA_DIRECTORY, SMILES_DIRECTORY, SMILES_DIRECTORY]
DATA_TYPES = ["FASTA", "SMILES", "SMILES"]
DATASET_NAMES = ["FASTA", "bace", "BBBP"]
NUM_SAMPLES = 50




def cv_and_train(doc_list, dataset, data_type, sample):
    cv = CountVectorize(doc_list=doc_list, data_type=data_type)

    embeddings = cv.embedding

    train_data, test_data, train_labels, test_labels = train_test_split(embeddings, labels, test_size=0.2,
                                                                        random_state=0)

    model_parameters = {
        "model_name": "experiment8_" + dataset + "_cv_nbc",
        "dataset_name": dataset
    }

    nbc = nbc_master_class(model_parameters, train_data=train_data, test_data=test_data, train_labels=train_labels,
                           test_labels=test_labels, load_model=None, embedding_type="cv")
    nbc.train()
    nbc.predict()
    nbc.evaluate()
    test_metrics = nbc.test_metrics

    return test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]


def tfidf_and_train(doc_list, dataset, data_type, sample):
    tfidf = TFIDF(doc_list=doc_list, data_type=data_type)

    embeddings = tfidf.embedding

    train_data, test_data, train_labels, test_labels = train_test_split(embeddings, labels, test_size=0.2,
                                                                        random_state=0)

    model_parameters = {
        "model_name": "experiment8_" + dataset + "_tfidf_nbc",
        "dataset_name": dataset
    }

    nbc = nbc_master_class(model_parameters, train_data=train_data, test_data=test_data, train_labels=train_labels,
                           test_labels=test_labels, load_model=None, embedding_type="tfidf")
    nbc.train()
    nbc.predict()
    nbc.evaluate()
    test_metrics = nbc.test_metrics

    return test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]


def word2vec_and_train(doc_list, dataset, data_type, sample):
    word2vec_model = word2vec(doc_list=doc_list, data_type=data_type)

    embeddings = word2vec_model.embedding

    train_data, test_data, train_labels, test_labels = train_test_split(embeddings, labels, test_size=0.2,
                                                                        random_state=0)

    model_parameters = {
        "model_name": "experiment8_" + dataset + "_word2vec_nbc",
        "dataset_name": dataset
    }

    nbc = nbc_master_class(model_parameters, train_data=train_data, test_data=test_data, train_labels=train_labels,
                           test_labels=test_labels, load_model=None, embedding_type="word2vec")
    nbc.train()
    nbc.predict()
    nbc.evaluate()
    test_metrics = nbc.test_metrics

    return test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]




experiment_folder = os.path.join(LOG_DIRECTORY, "experiment8")
if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

for i in range(len(DATA_LIST)):
    df = pd.read_csv(os.path.join(DATA_PATHS[i], DATA_LIST[i]))
    labels = df["classification"]

    if DATA_TYPES[i] == "FASTA":
        doc_list = df["sequence"].to_list()
    else:
        doc_list = df["sentence"].to_list()

    # CV
    metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                              index=range(NUM_SAMPLES))
    metrics_list = Parallel(n_jobs=5)(
        delayed(cv_and_train)(doc_list=doc_list, dataset=DATASET_NAMES[i], data_type=DATA_TYPES[i], sample=sample) for sample in tqdm(range(NUM_SAMPLES)))
    experiment_folder = os.path.join(LOG_DIRECTORY, "experiment8")
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)

    test_metrics_file_name = os.path.join(experiment_folder, "experiment8_" + DATASET_NAMES[i] + "_cv_nbc.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder, "experiment8_" + DATASET_NAMES[i] + "_cv_nbc_stats.csv")
    metrics_df.iloc[:NUM_SAMPLES] = metrics_list
    metrics_df.to_csv(test_metrics_file_name)
    metrics_df = metrics_df.apply(pd.to_numeric)
    statistics_df = metrics_df.describe(include='all')
    statistics_df.to_csv(metrics_statistics_file_name)

    # TFIDF
    metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                              index=range(NUM_SAMPLES))
    metrics_list = Parallel(n_jobs=5)(
        delayed(tfidf_and_train)(doc_list=doc_list, dataset=DATASET_NAMES[i], data_type=DATA_TYPES[i], sample=sample) for sample in tqdm(range(NUM_SAMPLES)))
    experiment_folder = os.path.join(LOG_DIRECTORY, "experiment8")
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)

    test_metrics_file_name = os.path.join(experiment_folder, "experiment8_" + DATASET_NAMES[i] + "_tfidf_nbc.csv")
    metrics_statistics_file_name = os.path.join(experiment_folder, "experiment8_" + DATASET_NAMES[i] + "_tfidf_nbc_stats.csv")
    metrics_df.iloc[:NUM_SAMPLES] = metrics_list
    metrics_df.to_csv(test_metrics_file_name)
    metrics_df = metrics_df.apply(pd.to_numeric)
    statistics_df = metrics_df.describe(include='all')
    statistics_df.to_csv(metrics_statistics_file_name)

    # word2vec for SMILES datasets
    if DATA_TYPES[i] != "FASTA":
        metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                                  index=range(NUM_SAMPLES))
        metrics_list = Parallel(n_jobs=4)(
            delayed(word2vec_and_train)(doc_list=doc_list, dataset=DATASET_NAMES[i], data_type=DATA_TYPES[i],
                                     sample=sample) for sample in tqdm(range(NUM_SAMPLES)))
        experiment_folder = os.path.join(LOG_DIRECTORY, "experiment8")
        if not os.path.exists(experiment_folder):
            os.mkdir(experiment_folder)

        test_metrics_file_name = os.path.join(experiment_folder, "experiment8_" + DATASET_NAMES[i] + "_word2vec_nbc.csv")
        metrics_statistics_file_name = os.path.join(experiment_folder,
                                                    "experiment8_" + DATASET_NAMES[i] + "_word2vec_nbc_stats.csv")
        metrics_df.iloc[:NUM_SAMPLES] = metrics_list
        metrics_df.to_csv(test_metrics_file_name)
        metrics_df = metrics_df.apply(pd.to_numeric)
        statistics_df = metrics_df.describe(include='all')
        statistics_df.to_csv(metrics_statistics_file_name)

print(0)