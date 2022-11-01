import pandas as pd
from src.paths import *
from src.datasets.pdb import PDB
from src.utilities import *
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from sklearn import metrics
from src.embeddings.LDA import LDA


pdb_list = read_list(os.path.join(LISTS_DIRECTORY, "pdb_list_5481.txt"))

receptor_list = [i.split("_")[0] + "_" + i.split("_")[1] for i in pdb_list if len(i.split("_")) == 3]
ligand_list = [i.split("_")[0] + "_" + i.split("_")[2] for i in pdb_list if len(i.split("_")) == 3]

text_list = []
for pdb_code in tqdm(receptor_list):
    if os.path.isfile(os.path.join(PDB_RAW_DIRECTORY, pdb_code + ".pdb")):
        pdb = PDB(pdb_code)
        text_list.append(pdb.pdb_text)

for pdb_code in tqdm(ligand_list):
    if os.path.isfile(os.path.join(PDB_RAW_DIRECTORY, pdb_code + ".pdb")):
        pdb = PDB(pdb_code)
        text_list.append(pdb.pdb_text)


dim = 10

lda_model = LDA(doc_list=text_list, data_type="PDB", num_topics=dim)
lda_model.save_embedding(os.path.join(EMBEDDING_DIRECTORY, "PDB_" + str(dim) + "_lda.embeddding.npy"))


pairs = len(receptor_list)
samples = lda_model.embedding.shape[0]
topics = lda_model.embedding.shape[1]

distance_array = np.empty([pairs, pairs])
coeffictient_array = np.empty([pairs, pairs])


def bhattacharyya_coeff(arr1, arr2):
    return np.sum(np.sqrt(arr1 * arr2), axis=1)


def bhattacharyya_dist(arr1, arr2):
    coeff = bhattacharyya_coeff(arr1, arr2)
    coeff[coeff == 0] = 0.000001
    return -np.log(coeff)

receptor_embeddings = lda_model.embedding[:pairs, :]
ligand_embeddings = lda_model.embedding[pairs:, :]


for i in tqdm(range(int(pairs))):
    distance_array[i, :] = bhattacharyya_dist(receptor_embeddings[i, :], ligand_embeddings)
    # coeffictient_array[i, :] = bhattacharyya_coeff(lda_model.embedding[i], lda_model.embedding)

diagonal = np.diagonal(distance_array)
true_mean = np.mean(diagonal)
exclude_diag = distance_array[~np.isnan(distance_array)]

proportion_list = []
flattened_distance_array = distance_array.reshape(-1)

for i in tqdm(range(1000000)):
    distance_sample = np.random.choice(flattened_distance_array, pairs, replace=True)
    proportion_list.append(np.mean(distance_sample))

print(np.mean(distance_array))
print(true_mean)
s = np.sqrt(((flattened_distance_array.shape[0]-1) * np.std(flattened_distance_array) + (diagonal.shape[0] * np.std(diagonal))) / (diagonal.shape[0] + flattened_distance_array.shape[0] + 2))
print((np.mean(flattened_distance_array)-true_mean)/s)

count = [i for i in proportion_list if i < true_mean]

plt.hist(proportion_list)
plt.title("p-value:" + str(len(count)/1000000))
plt.axvline(true_mean, color='r', linewidth=2, label="Mean distance between known interactions")
plt.axvline(np.mean(distance_array), color='black', linewidth=2, label="Mean distance between all interactions")

plt.legend()
plt.savefig(os.path.join(LOG_DIRECTORY, "ppi_histogram.png"), format='png', dpi=120)
# plt.show()
plt.close()

idx = [i for i in range(int(pairs))]
test_distances = exclude_diag.reshape(pairs, -1)


prob_list = []
dummy_prob = []
for i in tqdm(idx):
    filtered = test_distances[i, :][test_distances[i, :] > diagonal[i]]
    prob_list.append(filtered.shape[0]/test_distances[i, :].shape[0])
    random_idx = np.random.choice([j for j in idx if j != i], 1, replace=True)
    for j in random_idx:
        dummy_filter = test_distances[i, :][test_distances[i, :] > test_distances[i, j]]
        dummy_prob.append(dummy_filter.shape[0] / test_distances[i, :].shape[0])

predictions = prob_list + dummy_prob
labels = [1]*len(prob_list) + [0] * len(dummy_prob)

print(metrics.roc_auc_score(labels, predictions))
