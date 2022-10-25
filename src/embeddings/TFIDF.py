"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 August 2022
@Version  : 0.1
@Desc     :
"""

from gensim.models import TfidfModel
from gensim.corpora.dictionary import Dictionary
import os
import pandas as pd
import numpy as np
from src.paths import *

class TFIDF():
    """
    Class for TFIDF with methods to calculate, save and load embeddings

    Args:
        doc_list: list: str
        data_type: str
    """
    def __init__(self, doc_list, data_type):
        self.doc_list = doc_list

        self.docs_tokenised = [[d for d in doc.split(' ') if d] for doc in doc_list]
        if data_type in ["FASTA", "PDB"]:
            self.docs_tokenised = [[d for d in doc if d != "X"] for doc in self.docs_tokenised]

        self.dictionary = Dictionary(self.docs_tokenised)

        self.bow = [self.dictionary.doc2bow(text) for text in self.docs_tokenised]

        self.model = TfidfModel(self.bow)

        self.calculate_embedding()


    def calculate_embedding(self):
        self.embedding = np.zeros((len(self.bow), len(self.dictionary.items())))
        tfidf_matrix = self.model[self.bow]

        for i in range(len(self.bow)):
            for index, tfidf in tfidf_matrix[i]:
                self.embedding[i, index] = tfidf

        self.embedding_initialized = True


    def save_tfidf_model(self, save_path):

        self.model.save(save_path)


    def get_embedding(self):

        return self.embedding


    def save_embedding(self, file_name):
        if self.embedding_initialized:
            version = np.array([1])
            with open(file_name, "wb") as f:
                # Version
                np.save(f, version)
                np.save(f, self.embedding)


    @staticmethod
    def load_embedding(file_name):
        with open(file_name, "rb") as f:
            # Version
            version = np.load(f, allow_pickle=True)
            embedding = np.load(f, allow_pickle=True)

        return embedding


def create_tfidf_embeddings(data_types, data_paths, data_list, dataset_names):
    for i in range(len(data_list)):
        df = pd.read_csv(os.path.join(data_paths[i], data_list[i]))

        if data_types[i] == "FASTA":
            doc_list = df["sequence"].to_list()
        else:
            doc_list = df["sentence"].to_list()

        lda_model = TFIDF(doc_list=doc_list, data_type=data_types[i])

        experiment_models_path = os.path.join(MODEL_DIRECTORY, "embedding_models")
        if not os.path.exists(experiment_models_path):
            os.mkdir(experiment_models_path)

        run_model_path = os.path.join(experiment_models_path, dataset_names[i] + "_tfidf")
        if not os.path.exists(run_model_path):
            os.mkdir(run_model_path)

        lda_model.save_tfidf_model(os.path.join(run_model_path, dataset_names[i] + "_tfidf.model"))
        lda_model.save_embedding(os.path.join(EMBEDDING_DIRECTORY, dataset_names[i] + "_tfidf.embeddding.npy"))
