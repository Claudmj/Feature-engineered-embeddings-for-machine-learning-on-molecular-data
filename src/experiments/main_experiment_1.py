"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 14 September 2022
@Version  : 0.1
@Desc     : This file will be used to test classes and functions in development phase.
"""

from src.embeddings.LDA import LDA
from src.embeddings.word2vec import word2vec
from src.datasets.fasta import filter_and_clean_fasta
from src.datasets.smiles import filter_and_clean_smiles
from src.modelling.support_vector_machine_classifier import svm_master_class
from sklearn.model_selection import train_test_split
from src.paths import *
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


# Initialize what processes should be run and initialize variables
CLEAN_DATASET = False
DIM_LIST = [10, 20, 50, 100]
DATA_LIST = ["cleaned_binary.csv", "cleaned_bace.csv", "cleaned_BBBP.csv"]
DATA_PATHS = [FASTA_DIRECTORY, SMILES_DIRECTORY, SMILES_DIRECTORY]
DATA_TYPES = ["FASTA", "SMILES", "SMILES"]
DATASET_NAMES = ["FASTA", "bace", "BBBP"]
NUM_SAMPLES = 50



# def embed_and_train(dataset, data_type, num_samples, dim):
#         metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"], index=range(num_samples))
#         for sample in range(num_samples):
#
#             lda_model = LDA(doc_list=doc_list, data_type=data_type, num_topics=dim)
#
#             embeddings = lda_model.embedding
#
#             train_data, test_data, train_labels, test_labels = train_test_split(embeddings, labels, test_size=0.2, random_state=0)
#
#             model_parameters = {
#                 "model_name": "experiment1_" + dataset + "_" + str(dim) + "_lda_svm",
#                 "dataset_name": dataset
#             }
#
#             svm = svm_master_class(model_parameters, train_data=train_data, test_data=test_data, train_labels=train_labels, test_labels=test_labels, load_model=None)
#             svm.train()
#             svm.predict()
#             svm.evaluate()
#             test_metrics = svm.test_metrics
#             metrics_df.iloc[sample] = [test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]]
#
#
#         test_metrics_file_name = os.path.join(experiment_folder, svm.model_name + ".csv")
#         metrics_statistics_file_name = os.path.join(experiment_folder, svm.model_name + "_stats.csv")
#         metrics_df.to_csv(test_metrics_file_name)
#         metrics_df = metrics_df.apply(pd.to_numeric)
#         statistics_df = metrics_df.describe(include='all')
#         statistics_df.to_csv(metrics_statistics_file_name)



def embed_and_train2(doc_list, dataset, data_type, dim, sample):
    lda_model = LDA(doc_list=doc_list, data_type=data_type, num_topics=dim)

    embeddings = lda_model.embedding

    train_data, test_data, train_labels, test_labels = train_test_split(embeddings, labels, test_size=0.2,
                                                                        random_state=0)

    model_parameters = {
        "model_name": "experiment1_" + dataset + "_" + str(dim) + "_lda_svm",
        "dataset_name": dataset
    }

    svm = svm_master_class(model_parameters, train_data=train_data, test_data=test_data, train_labels=train_labels,
                           test_labels=test_labels, load_model=None)
    svm.train()
    svm.predict()
    svm.evaluate()
    test_metrics = svm.test_metrics

    return test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"], test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]
    # metrics_df.iloc[sample] = [test_metrics["Accuracy"], test_metrics["AUC"], test_metrics["Precision"],
    #                            test_metrics["Recall"], test_metrics["F1"], test_metrics["JI"], test_metrics["MCC"]]


experiment_folder = os.path.join(LOG_DIRECTORY, "experiment1")
if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

for i in tqdm(range(len(DATA_LIST))):
    df = pd.read_csv(os.path.join(DATA_PATHS[i], DATA_LIST[i]))
    labels = df["classification"]

    if DATA_TYPES[i] == "FASTA":
        doc_list = df["sequence"].to_list()
    else:
        doc_list = df["sentence"].to_list()

    for dim in DIM_LIST:
        metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "Precision", "Recall", "F1", "JI", "MCC"],
                                  index=range(NUM_SAMPLES))
        metrics_list = Parallel(n_jobs=5)(
            delayed(embed_and_train2)(doc_list=doc_list, dataset=DATASET_NAMES[i], data_type=DATA_TYPES[i], dim=dim,
                                      sample=sample) for sample in range(NUM_SAMPLES))
        experiment_folder = os.path.join(LOG_DIRECTORY, "experiment1")
        if not os.path.exists(experiment_folder):
            os.mkdir(experiment_folder)

        test_metrics_file_name = os.path.join(experiment_folder, "experiment1_" + DATASET_NAMES[i] + "_" + str(dim) + "_lda_svm.csv")
        metrics_statistics_file_name = os.path.join(experiment_folder, "experiment1_" + DATASET_NAMES[i] + "_" + str(dim) + "_lda_svm_stats.csv")
        metrics_df.iloc[:NUM_SAMPLES] = metrics_list
        metrics_df.to_csv(test_metrics_file_name)
        metrics_df = metrics_df.apply(pd.to_numeric)
        statistics_df = metrics_df.describe(include='all')
        statistics_df.to_csv(metrics_statistics_file_name)

print(0)