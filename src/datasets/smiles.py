"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     :
"""

import pandas as pd
import os
from src.paths import *
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec


def filter_and_clean_smiles(file):
    """
    Cleans and divides molecule into substructures. Each substructure is then encoded using a Morgan fingerprint.
     These molecular sentences are then merged with classification labels.

    Args:
        file: str
    Returns:
        none.  The merged and cleaned dataframe is saved in csv format.
    """
    df = pd.read_csv(os.path.join(SMILES_DIRECTORY, file))
    df = df.drop_duplicates()
    df = df.dropna()

    df["mol"] = df["smiles"].apply(lambda x: Chem.MolFromSmiles(x, sanitize=False))
    df["problems"] = df["mol"].apply(lambda x: len(Chem.DetectChemistryProblems(x)))
    df = df[df["problems"] == 0]
    df["mol"] = df["smiles"].apply(lambda x: Chem.MolFromSmiles(x, sanitize=True))

    sentence_df = pd.DataFrame()
    sentence_df["sentence"] = df.apply(lambda x: " ".join(MolSentence(mol2alt_sentence(x["mol"], 1)).sentence), axis=1)
    df["mol_sentence"] = df.apply(lambda x: MolSentence(mol2alt_sentence(x["mol"], 1)), axis=1)

    sentence_df["classification"] = df["classification"]
    molsentence_df = df[["classification", "mol_sentence"]]

    sentence_df.to_csv(os.path.join(SMILES_DIRECTORY, "cleaned_" + file))
    molsentence_df.to_csv(os.path.join(SMILES_DIRECTORY, "cleaned_mol_sentence_" + file))



# filter_and_clean_smiles("BBBP.csv")