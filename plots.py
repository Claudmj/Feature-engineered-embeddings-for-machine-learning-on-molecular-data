"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     : This file will be used to test classes and functions in development phase.
"""

import os
from src.paths import *
# from src.datasets.smiles import *
import pandas as pd
import numpy as np
from src.embeddings.word2vec import word2vec
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")
# sns.set(style="darkgrid")
sns.set_style("ticks")

def get_plot_df(dataset_list, dim_list, experiment, embedding_type, model):
    accuracy_list = []
    label_list = []
    color_list = []
    for set in dataset_list:
        for dim in dim_list:
            df = pd.read_csv(os.path.join(LOG_DIRECTORY, f"{experiment}/{experiment}_{set}_{dim}_{embedding_type}_{model}.csv"))
            accuracy_list += df["Accuracy"].tolist()
            label_list += [f"{set} {dim}"]*50
            color_list += [f"{set}"]*50

    return pd.DataFrame({"Accuracy": accuracy_list, "Dataset and dimension": label_list, "Dataset": color_list})


def get_df_and_plot(dataset_list, dim_list, experiment, embedding_type, model):
    accuracy_list = []
    label_list = []
    color_list = []
    for set in dataset_list:
        for dim in dim_list:
            df = pd.read_csv(os.path.join(LOG_DIRECTORY, f"{experiment}/{experiment}_{set}_{dim}_{embedding_type}_{model}.csv"))
            accuracy_list += df["Accuracy"].tolist()
            label_list += [f"{set} {dim}"]*50
            color_list += [f"{set}"]*50

    plt.figure(figsize=(15, 8))
    df = pd.DataFrame({"Accuracy": accuracy_list, "Dataset and dimension": label_list, "Dataset": color_list})
    ax = sns.boxplot(x=df["Dataset and dimension"], y=df["Accuracy"], hue=df["Dataset"], dodge=False, width=.5)
    plt.grid()
    # plt.show()
    fig = ax.get_figure()
    fig.savefig(os.path.join(LOG_DIRECTORY, f"{experiment}/{experiment}_boxplot.png"), format='png', dpi=120)


def get_df_and_plot2(dataset_list, experiment, embedding_list, model):
    embedding_dict = {"lda": "LDA", "cv": "CVec", "tfidf": "TFIDF", "word2vec": "word2vec"}
    accuracy_list = []
    label_list = []
    color_list = []
    for set in dataset_list:
        for embedding in embedding_list:
            df = pd.read_csv(os.path.join(LOG_DIRECTORY, f"{experiment}/{experiment}_{set}_{embedding}_{model}.csv"))
            accuracy_list += df["Accuracy"].tolist()
            label_list += [f"{set} {embedding_dict[embedding]}"]*50
            color_list += [f"{set}"]*50

    plt.figure(figsize=(15, 8))
    df = pd.DataFrame({"Accuracy": accuracy_list, "Dataset and embedding": label_list, "Dataset": color_list})
    ax = sns.boxplot(x=df["Dataset and embedding"], y=df["Accuracy"], hue=df["Dataset"], dodge=False, width=.5)
    plt.grid()
    plt.show()
    fig = ax.get_figure()
    fig.savefig(os.path.join(LOG_DIRECTORY, f"{experiment}/{experiment}_boxplot.png"), format='png', dpi=120)


def plot_experiment10(dataset_list, experiment_list, embedding_list, model_list):
    embedding_dict = {"lda": "LDA", "cv": "CVec", "tfidf": "TFIDF", "word2vec": "word2vec"}
    model_dict = {"svm": "SVM", "nbc": "NBC", "nn": "Neural network"}
    accuracy_list = []
    label_list = []
    color_list = []
    for i in range(len(dataset_list)):
        set = dataset_list[i]
        iteration_embeddings = embedding_list[i]
        for i in range(len(model_list)):
            df = pd.read_csv(os.path.join(LOG_DIRECTORY, f"{experiment_list[i]}/{experiment_list[i]}_{set}_{iteration_embeddings[i]}_{model_list[i]}.csv"))
            accuracy_list += df["Accuracy"].tolist()
            label_list += [f"{set} {model_dict[model_list[i]]} \n {embedding_dict[iteration_embeddings[i]]}"]*50
            color_list += [f"{set}"]*50

    plt.figure(figsize=(15, 8))
    df = pd.DataFrame({"Accuracy": accuracy_list, "Dataset with model and embedding": label_list, "Dataset": color_list})
    ax = sns.boxplot(x=df["Dataset with model and embedding"], y=df["Accuracy"], hue=df["Dataset"], dodge=False, width=.5)
    plt.grid()
    plt.show()
    fig = ax.get_figure()
    fig.savefig(os.path.join(LOG_DIRECTORY, f"experiment10_boxplot.png"), format='png', dpi=120)



get_df_and_plot(dataset_list=["bace", "BBBP", "FASTA"], dim_list=[10, 20, 50, 100], experiment="experiment1", embedding_type="lda", model="svm")
get_df_and_plot(dataset_list=["bace", "BBBP", "FASTA"], dim_list=[10, 20, 50, 100], experiment="experiment2", embedding_type="lda", model="nbc")
get_df_and_plot(dataset_list=["bace", "BBBP", "FASTA"], dim_list=[10, 20, 50, 100], experiment="experiment3", embedding_type="lda", model="nn")
get_df_and_plot(dataset_list=["FASTA"], dim_list=[100, 200, 300], experiment="experiment4", embedding_type="word2vec", model="svm")
get_df_and_plot(dataset_list=["FASTA"], dim_list=[100, 200, 300], experiment="experiment5", embedding_type="word2vec", model="nbc")
get_df_and_plot(dataset_list=["FASTA"], dim_list=[100, 200, 300], experiment="experiment6", embedding_type="word2vec", model="nn")
get_df_and_plot2(dataset_list=["bace", "BBBP", "FASTA"], experiment="experiment7", embedding_list=["cv", "tfidf", "lda", "word2vec"], model="svm")
get_df_and_plot2(dataset_list=["bace", "BBBP", "FASTA"], experiment="experiment8", embedding_list=["cv", "tfidf", "lda", "word2vec"], model="nbc")
get_df_and_plot2(dataset_list=["bace", "BBBP", "FASTA"], experiment="experiment9", embedding_list=["cv", "tfidf", "lda", "word2vec"], model="nn")

experiment_list = ["experiment7", "experiment8", "experiment9"]
embedding_list = [["tfidf", "cv", "cv"], ["tfidf", "cv", "cv"], ["cv", "cv", "cv"]]
model_list = ["svm", "nbc", "nn"]
plot_experiment10(dataset_list=["bace", "BBBP", "FASTA"], experiment_list=experiment_list, embedding_list=embedding_list, model_list=model_list)
