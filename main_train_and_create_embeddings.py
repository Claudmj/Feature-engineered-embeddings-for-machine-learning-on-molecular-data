"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 14 September 2022
@Version  : 0.1
@Desc     : This file will be used to test classes and functions in development phase.
"""


from src.embeddings.LDA import create_lda_embeddings
from src.embeddings.CountVectorize import create_cv_embeddings
from src.embeddings.TFIDF import create_tfidf_embeddings
from src.embeddings.word2vec import create_word2vec_embeddings
from src.paths import *
from src.datasets.fasta import filter_and_clean_fasta
from src.datasets.smiles import filter_and_clean_smiles

CREATE_LDA_EMBEDDINGS = True
CREATE_WORD2VEC_EMBEDDINGS = True
CREATE_CV_EMBEDDINGS = True
CREATE_TFIDF_EMBEDDINGS = True

CLEAN_DATASET = True
LDA_DIM_LIST = [10, 20, 50, 100]
WORD2VEC_DIM_LIST = [10, 20, 50, 100]
WORD2VEC_DATA_LIST = ["cleaned_binary.csv", "cleaned_mol_sentence_bace.csv", "cleaned_mol_sentence_BBBP.csv"]
DATA_LIST = ["cleaned_binary.csv", "cleaned_bace.csv", "cleaned_BBBP.csv"]
DATA_PATHS = [FASTA_DIRECTORY, SMILES_DIRECTORY, SMILES_DIRECTORY]
DATA_TYPES = ["FASTA", "SMILES", "SMILES"]
DATASET_NAMES = ["FASTA", "bace", "BBBP"]


if CLEAN_DATASET:
    filter_and_clean_fasta()
    filter_and_clean_smiles("bace.csv")
    filter_and_clean_smiles("BBBP.csv")

if CREATE_LDA_EMBEDDINGS:
    create_lda_embeddings(data_types=DATA_TYPES, data_paths=DATA_PATHS, data_list=DATA_LIST, dataset_names=DATASET_NAMES, dim_list=LDA_DIM_LIST)


if CREATE_CV_EMBEDDINGS:
    create_cv_embeddings(data_types=DATA_TYPES, data_paths=DATA_PATHS, data_list=DATA_LIST, dataset_names=DATASET_NAMES)


if CREATE_TFIDF_EMBEDDINGS:
    create_tfidf_embeddings(data_types=DATA_TYPES, data_paths=DATA_PATHS, data_list=DATA_LIST, dataset_names=DATASET_NAMES)


if CREATE_WORD2VEC_EMBEDDINGS:
    create_word2vec_embeddings(data_types=DATA_TYPES, data_paths=DATA_PATHS, data_list=WORD2VEC_DATA_LIST, dataset_names=DATASET_NAMES, dim_list=WORD2VEC_DIM_LIST)