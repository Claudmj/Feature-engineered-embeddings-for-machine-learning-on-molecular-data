"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     : This file will be used to test classes and functions in development phase.
"""

from gensim.models import Word2Vec
from src.paths import *
import pandas as pd
import os
from src.paths import *
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor




class word2vec():
    """
    Class for word2vec with methods to calculate, save and load embeddings

    Args:
        doc_list: list: str
        data_type: str
        dimension: int
    """
    def __init__(self, doc_list, data_type, dimension=100):
        self.embedding_initialized = False
        self.doc_list = doc_list
        self.dimension = dimension

        if data_type == "FASTA":
            self.unseen = None
            self.doc_list = [[d for d in doc if d != "X"] for doc in self.doc_list]
            self.model = Word2Vec(self.doc_list, window=5, vector_size=self.dimension)
            self.calculate_embedding()

        elif data_type == "SMILES":
            self.unseen = "UNK"
            self.model = Word2Vec.load(os.path.join(EMBEDDING_MODEL_DIRECTORY, "model_300dim.pkl"))
            self.calculate_embedding()

        elif data_type == "PDB":
            self.unseen = None
            self.doc_list = [[d for d in doc if d != "X"] for doc in self.doc_list]
            self.model = Word2Vec(self.doc_list, window=5, vector_size=self.dimension, min_count=100, workers=7)
            self.calculate_embedding_distributed()


    def save_word2vec_model(self, save_path):

        self.model.save(save_path)


    def calculate_embedding(self):
        keys = set(self.model.wv.index_to_key)
        vec = []

        if self.unseen is not None:
            unseen_vec = self.model.wv.get_vector(self.unseen)

        for sentence in self.doc_list:
            if self.unseen is not None:
                vec.append(sum([self.model.wv.get_vector(y) if y in set(sentence) & keys
                                else unseen_vec for y in sentence]))
            else:
                vec.append(sum([self.model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & keys]))

        self.embedding = np.array(vec)
        self.embedding_initialized = True


    def vec_append_distributed(self, sentence, vec, keys):
        if self.unseen is not None:
            vec.append(sum([self.model.wv.get_vector(y) if y in set(sentence) & keys
                            else unseen_vec for y in sentence]))
        else:
            vec.append(sum([self.model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & keys]))


    def vec_append_distributed2(self, sentence):
        if self.unseen is not None:
            res = sum([self.model.wv.get_vector(y) if y in set(sentence) & self.keys
                            else self.unseen_vec for y in sentence])
        else:
            res = sum([self.model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & self.keys])

        return res



    def calculate_embedding_distributed(self):
        self.keys = set(self.model.wv.index_to_key)
        vec = []

        if self.unseen is not None:
            self.unseen_vec = self.model.wv.get_vector(self.unseen)

        # Parallel(n_jobs=7)(delayed(self.vec_append_distributed)(i, vec, keys) for i in tqdm(self.doc_list))
        with ProcessPoolExecutor(max_workers=4) as executor:
            for r in tqdm(executor.map(self.vec_append_distributed2, self.doc_list)):
                vec.append(r)

        self.embedding = np.array(vec)
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


def train_and_save_embeddings(doc_list, data_type, dataset_name, dim=None):
        word2vec_model = word2vec(doc_list=doc_list, data_type=data_type, dimension=dim)

        experiment_models_path = os.path.join(MODEL_DIRECTORY, "embedding_models")
        if not os.path.exists(experiment_models_path):
            os.mkdir(experiment_models_path)

        run_model_path = os.path.join(experiment_models_path, dataset_name + "_" + str(dim) + "_word2vec")
        if not os.path.exists(run_model_path):
            os.mkdir(run_model_path)

        word2vec_model.save_word2vec_model(
            os.path.join(run_model_path, dataset_name + "_" + str(dim) + "_word2vec.model"))
        word2vec_model.save_embedding(
            os.path.join(EMBEDDING_DIRECTORY, dataset_name + "_" + str(dim) + "_word2vec.embeddding.npy"))


def create_word2vec_embeddings(data_types, data_paths, data_list, dataset_names, dim_list):
    for i in range(len(data_list)):
        df = pd.read_csv(os.path.join(data_paths[i], data_list[i]))

        if data_types[i] == "FASTA":
            doc_list = df["sequence"].to_list()
            for dim in dim_list:
                train_and_save_embeddings(doc_list=doc_list, data_type=data_types[i], dataset_name=dataset_names[i], dim=dim)

        else:
            doc_list = df["mol_sentence"].to_list()
            train_and_save_embeddings(doc_list=doc_list, data_type=data_types[i], dataset_name=dataset_names[i], dim="mol2vec")