"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 August 2022
@Version  : 0.1
@Desc     :
"""
import os

from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
import pandas as pd
import os
from src.paths import *


class LDA():
    """
    Class for LDA with methods to calculate, save and load embeddings

    Args:
        doc_list: list: str
        data_type: str
        num_topics: int
    """
    def __init__(self, doc_list, data_type, num_topics):
        self.embedding_initialized = False
        self.doc_list = doc_list
        self.num_topics = num_topics

        self.docs_tokenised = [[d for d in doc.split(' ') if d] for doc in doc_list]
        if data_type in ["FASTA", "PDB"]:
            self.docs_tokenised = [[d for d in doc if d != "X"] for doc in self.docs_tokenised]

        self.dictionary = Dictionary(self.docs_tokenised)

        self.bow = [self.dictionary.doc2bow(text) for text in self.docs_tokenised]

        self.model = LdaModel(corpus=self.bow,
                                        num_topics=self.num_topics,
                                        id2word=self.dictionary)
        self.calculate_embedding()

    def save_lda_model(self, save_path):

        self.model.save(save_path)


    def calculate_embedding(self):
        self.embedding = np.zeros((len(self.bow), self.num_topics))

        for i in range(len(self.bow)):
            row = self.model.get_document_topics(self.bow[i])
            for topic, prob in row:
                self.embedding[i, topic] = prob

        self.embedding_initialized = True


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


    # def lda_tfidf(self):
    #     self.tfidf_model = TfidfModel(self.bow)
    #     self.tfidf_vector = self.model[self.bow]
    #
    #     self.lda_tfidf_model = LdaModel(corpus=self.tfidf_vector,
    #                      num_topics=self.num_topics,
    #                      id2word=self.dictionary)
    #
    #
    # def save_lda_tfidf_model(self, save_path):
    #     tfidf_path = os.path.join(save_path, "tfidf")
    #     lda_tfidf_path = os.path.join(save_path, "lda_tfidf")
    #
    #     if !os.path.exists(save_path):
    #         os.mkdir(save_path)
    #
    #     if !os.path.exists(tfidf_path):
    #         os.mkdir(tfidf_path)
    #
    #     if !os.path.exists(lda_tfidf_path):
    #         os.mkdir(lda_tfidf_path)
    #
    #     self.tfidf_model.save(tfidf_path)
    #     self.lda_tfidf_model.save(lda_tfidf_path)



def create_lda_embeddings(data_types, data_paths, data_list, dataset_names, dim_list):
    for i in range(len(data_list)):
        df = pd.read_csv(os.path.join(data_paths[i], data_list[i]))

        if data_types[i] == "FASTA":
            doc_list = df["sequence"].to_list()
        else:
            doc_list = df["sentence"].to_list()

        for dim in dim_list:
            lda_model = LDA(doc_list=doc_list, data_type=data_types[i], num_topics=dim)

            experiment_models_path = os.path.join(MODEL_DIRECTORY, "embedding_models")
            if not os.path.exists(experiment_models_path):
                os.mkdir(experiment_models_path)

            run_model_path = os.path.join(experiment_models_path, dataset_names[i] + "_" + str(dim) + "_lda")
            if not os.path.exists(run_model_path):
                os.mkdir(run_model_path)

            lda_model.save_lda_model(os.path.join(run_model_path, dataset_names[i] + "_" + str(dim) + "_lda.model"))
            lda_model.save_embedding(os.path.join(EMBEDDING_DIRECTORY, dataset_names[i] + "_" + str(dim) + "_lda.embeddding.npy"))
