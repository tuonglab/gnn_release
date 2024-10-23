# Utilising Machine Learning to classify cancer using Paedatric Immune Repertoire

This project utilises Graph Neural Network as the Deep Learning Framework. The immune repertoire will be sampled and transformed under a few preprocessing steps.

The input requires raw CDR3 sequences collected and it must be from TCR-Beta chain of these immune repertoire.

1) Install required packages at `requirements.txt`
2) Retrieve the pdb files from Alphafold output for your CDR3 sequences
3) In `graph_generation` folder, run `create_edgelist.py` and run `process.py` to obtain the graphs for your CDR3 sequences
4) Now in the project root, run `test.py` to predict the results

To retrain the model on your own sequences,

run `train.py`

To test the inhouse data sample, run `process.py` on the folder path `/scratch/project/tcr_ml/gnn_release/test_data_v2/cancer/raw` and `/scratch/project/tcr_ml/gnn_release/test_data_v2/control/raw`

Then run test.py on the corresponding processed folder generated