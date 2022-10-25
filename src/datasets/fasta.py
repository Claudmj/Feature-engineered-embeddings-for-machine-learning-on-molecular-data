"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     :
"""

from src.paths import *
import os
import pandas as pd
import re


def filter_and_clean_fasta():
    """
    Cleans and merges FASTA sequence data with classification labels. The data is also filtered for Proteins sequences.

    Args:
        none
    Returns:
        none.  The merged, filtered and clean dataframe is saved in csv format.
    """

    # Read in data
    sequence_df = pd.read_csv(os.path.join(FASTA_DIRECTORY, "pdb_data_seq.csv"), usecols=["structureId", "sequence", "macromoleculeType", "chainId"])
    class_df = pd.read_csv(os.path.join(FASTA_DIRECTORY, "pdb_data_no_dups.csv"), usecols=["structureId", "classification", "macromoleculeType", "residueCount", "structureMolecularWeight", "densityMatthews", "densityPercentSol", "phValue"])

    sequence_df = sequence_df.drop_duplicates(subset=['sequence'])
    class_df = class_df.drop_duplicates()

    # Filter for proteins
    sequence_df = sequence_df[sequence_df.macromoleculeType == 'Protein']
    class_df = class_df[class_df.macromoleculeType == 'Protein']

    # Merge dataframe labels and sequence
    joined_df = pd.merge(sequence_df, class_df, on='structureId')
    joined_df = joined_df.reset_index()
    joined_df = joined_df.drop(["macromoleculeType_y"], axis=1)

    # Select only top 5 categories
    top2 = joined_df.classification.value_counts(ascending=False)[0:2]
    joined_df = joined_df[joined_df['classification'].isin(top2.index.to_list())]

    # Drop missing values
    joined_df = joined_df.dropna()

    # Create new ID with chain
    joined_df["protein_id"] = joined_df["structureId"] + "_" + joined_df["chainId"]

    # Drop duplicates
    joined_df = joined_df.drop_duplicates()

    # Space out residues in sequence data
    joined_df["sequence"] = joined_df["sequence"].str.upper()
    joined_df["sequence"] = joined_df["sequence"].apply(lambda x: re.sub('([A-Z])', r' \1', x))

    # Make labels numerical
    topics = top2.index.to_list()
    joined_df["classification"] = joined_df["classification"].apply(topics.index)

    # Save cleaned data
    joined_df.to_csv(os.path.join(FASTA_DIRECTORY, "cleaned_binary.csv"))
