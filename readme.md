# Feature engineered embeddings for machine learning on molecular data
Count vectorization, term frequency-inverse document frequency, latent Dirichlet allocation and word2vec are used to create embedding vectors of molecular text data in the form of protein sequences and SMILES strings.

A support vector machine classifier, naive Bayes classifier and simple neural network classifier are used to investigate the use of these embeddings as featrues. We predict the family of proteins using the protein sequences, blood-brain barrier penetration(permeability) and qualitative binary binding results for a set of inhibitors of human $\beta$-secretase 1(BACE-1).

We also investigate the use of the LDA embeddings in differentiating between proteins that are in a known protein-protein interaction and those that are not.

# Installation instructions

An Anaconda environment with Python 3.9 was used. Pull the project and navigate to the project folder:
- pip install -r requirements.txt


# Running
There is a main file for each of the different experiments in `src/experiments`. There is also a main file to run data cleaning, preprocessing and to create the embeddings.
