"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 14 September 2022
@Version  : 0.1
@Desc     : This file will be used to test classes and functions in development phase.
"""

from src.embeddings.word2vec import word2vec
from src.modelling.naive_bayes_classifier import nbc_master_class
from sklearn.model_selection import train_test_split
from src.paths import *
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


# Initialize what processes should be run and initialize variables
CLEAN_DATASET = False
DIM_LIST = [100, 200, 300]
DATA_LIST = ["cleaned_binary.csv"]
DATA_PATHS = [FASTA_DIRECTORY]
DATA_TYPES = ["FASTA"]
DATASET_NAMES = ["FASTA"]
NUM_SAMPLES = 50




def embed_and_train(doc_list, dataset, data_type, dim, sample):
    word2vec_model = word2vec(doc_list=doc_list, data_type=data_type, dimension=dim)

    embeddings = word2vec_model.embedding

    train_data, test_data, train_labels, test_labels = train_test_split(embeddings, labels, test_size=0.2,
                                                                        random_state=0)

    model_parameters = {
        "model_name": "experiment5_" + dataset + "_" + str(dim) + "_word2vec_nbc",
        "dataset_name": dataset
    }

    nbc = nbc_master_class(model_parameters, train_data=train_data, test_data=test_data, train_labels=train_labels,
                           test_labels=test_labels, embedding_type="word2vec", load_model=None)
    nbc.train()
    nbc.predict()
    nbc.evaluate()
    test_metrics = nbc.test_metrics

    return test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]



experiment_folder = os.path.join(LOG_DIRECTORY, "experiment5")
if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

for i in range(len(DATA_LIST)):
    df = pd.read_csv(os.path.join(DATA_PATHS[i], DATA_LIST[i]))
    labels = df["classification"]

    doc_list = df["sequence"].to_list()

    for dim in tqdm(DIM_LIST):
        metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                                  index=range(NUM_SAMPLES))
        metrics_list = Parallel(n_jobs=7)(
            delayed(embed_and_train)(doc_list=doc_list, dataset=DATASET_NAMES[i], data_type=DATA_TYPES[i], dim=dim,
                                      sample=sample) for sample in range(NUM_SAMPLES))
        experiment_folder = os.path.join(LOG_DIRECTORY, "experiment5")
        if not os.path.exists(experiment_folder):
            os.mkdir(experiment_folder)

        test_metrics_file_name = os.path.join(experiment_folder, "experiment5_" + DATASET_NAMES[i] + "_" + str(dim) + "_word2vec_nbc.csv")
        metrics_statistics_file_name = os.path.join(experiment_folder, "experiment5_" + DATASET_NAMES[i] + "_" + str(dim) + "_word2vec_nbc_stats.csv")
        metrics_df.iloc[:NUM_SAMPLES] = metrics_list
        metrics_df.to_csv(test_metrics_file_name)
        metrics_df = metrics_df.apply(pd.to_numeric)
        statistics_df = metrics_df.describe(include='all')
        statistics_df.to_csv(metrics_statistics_file_name)

print(0)