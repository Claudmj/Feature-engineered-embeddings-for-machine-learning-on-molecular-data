import numpy as np
import pandas as pd
import os
from src.paths import *
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
from src.embeddings.CountVectorize import CountVectorize
from src.embeddings.TFIDF import TFIDF
from deepchem.molnet import load_bbbp, load_bace_classification


def load_and_embed_cv_ml(dataset):

    if dataset == "bace":
        data = load_bace_classification(splitter="scaffold")
    else:
        data = load_bbbp(splitter="scaffold")

    df_train = data[1][0].to_dataframe()
    df_train = df_train[["ids", "y"]]
    df_train = df_train.rename(columns={"ids": "smiles", "y": "classification"})
    df_train["set"] = "train"

    df_valid = data[1][1].to_dataframe()
    df_valid = df_valid[["ids", "y"]]
    df_valid = df_valid.rename(columns={"ids": "smiles", "y": "classification"})
    df_valid["set"] = "valid"

    df_test = data[1][2].to_dataframe()
    df_test = df_test[["ids", "y"]]
    df_test = df_test.rename(columns={"ids": "smiles", "y": "classification"})
    df_test["set"] = "test"

    combined = pd.concat([df_train, df_valid, df_test], ignore_index=True)

    df = combined.drop_duplicates()
    df = df.dropna()

    df["mol"] = df["smiles"].apply(lambda x: Chem.MolFromSmiles(x, sanitize=False))
    df["problems"] = df["mol"].apply(lambda x: len(Chem.DetectChemistryProblems(x)))
    df = df[df["problems"] == 0]
    df["mol"] = df["smiles"].apply(lambda x: Chem.MolFromSmiles(x, sanitize=True))

    sentence_df = pd.DataFrame()
    sentence_df["sentence"] = df.apply(lambda x: " ".join(MolSentence(mol2alt_sentence(x["mol"], 1)).sentence), axis=1)
    df["mol_sentence"] = df.apply(lambda x: MolSentence(mol2alt_sentence(x["mol"], 1)), axis=1)

    sentence_df["classification"] = df["classification"]
    sentence_df["set"] = df["set"]
    molsentence_df = df[["classification", "mol_sentence"]]

    sentence_df.to_csv(os.path.join(SMILES_DIRECTORY, "cleaned_" + dataset + "_ml.csv"))

    doc_list = sentence_df["sentence"].to_list()

    CV = CountVectorize(doc_list=doc_list, data_type="SMILES")

    CV.save_embedding(os.path.join(EMBEDDING_DIRECTORY, dataset + "_ml_cv.embeddding.npy"))


def load_and_embed_tfidf_ml(dataset):

    if dataset == "bace":
        data = load_bace_classification(splitter="scaffold")
    else:
        data = load_bbbp(splitter="scaffold")

    df_train = data[1][0].to_dataframe()
    df_train = df_train[["ids", "y"]]
    df_train = df_train.rename(columns={"ids": "smiles", "y": "classification"})
    df_train["set"] = "train"
    # df_train.to_csv(os.path.join(SMILES_DIRECTORY, "BBBP_train.csv"))

    df_valid = data[1][1].to_dataframe()
    df_valid = df_valid[["ids", "y"]]
    df_valid = df_valid.rename(columns={"ids": "smiles", "y": "classification"})
    df_valid["set"] = "valid"
    # df_valid.to_csv(os.path.join(SMILES_DIRECTORY, "BBBP_valid.csv"))

    df_test = data[1][2].to_dataframe()
    df_test = df_test[["ids", "y"]]
    df_test = df_test.rename(columns={"ids": "smiles", "y": "classification"})
    df_test["set"] = "test"

    combined = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    # combined.to_csv(os.path.join(SMILES_DIRECTORY, set + "_ml.csv"))

    df = combined.drop_duplicates()
    df = df.dropna()

    df["mol"] = df["smiles"].apply(lambda x: Chem.MolFromSmiles(x, sanitize=False))
    df["problems"] = df["mol"].apply(lambda x: len(Chem.DetectChemistryProblems(x)))
    df = df[df["problems"] == 0]
    df["mol"] = df["smiles"].apply(lambda x: Chem.MolFromSmiles(x, sanitize=True))

    sentence_df = pd.DataFrame()
    sentence_df["sentence"] = df.apply(lambda x: " ".join(MolSentence(mol2alt_sentence(x["mol"], 1)).sentence), axis=1)
    df["mol_sentence"] = df.apply(lambda x: MolSentence(mol2alt_sentence(x["mol"], 1)), axis=1)

    sentence_df["classification"] = df["classification"]
    sentence_df["set"] = df["set"]
    molsentence_df = df[["classification", "mol_sentence"]]

    sentence_df.to_csv(os.path.join(SMILES_DIRECTORY, "cleaned_" + dataset + "_ml.csv"))

    doc_list = sentence_df["sentence"].to_list()

    tfidf = TFIDF(doc_list=doc_list, data_type="SMILES")

    tfidf.save_embedding(os.path.join(EMBEDDING_DIRECTORY, dataset + "_ml_tfidf.embeddding.npy"))





