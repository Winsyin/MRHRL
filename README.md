# MRHRL: Multi-Relational Hypergraph Representation Learning for Predicting circRNA-miRNA Associations
# Requirements
pytorch 1.9.0

torch-geometric 2.1.0

# File Annotation
1. CMA: the raw data
2. 5fold_CV: the training set, validation set and the test set in the 5-fold cross validation experiment
3. hyg: constructed hypergraph
4. similarity: molecular sequence similarity

# How to train the MRHRL model
You can train the model of 5-fold cross-validation with a very simple way by the command blow:  

python main.py
