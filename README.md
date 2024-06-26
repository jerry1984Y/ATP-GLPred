## ATP-GLPred
ATP-GLPred is a predictor for predicting protein-ATP binding residues 


To address the issue of extracting discriminative features from protein sequences in ligand binding site prediction, a method named ATP-Pred is proposed based on large language models. This method generates high-quality protein sequence features using a large language model and utilizes a multi-scale convolutional network to extract local features, effectively addressing the challenge of window adaptability faced by traditional methods that rely on fixed-size sliding windows for feature processing. Furthermore, this method employs self-attention mechanisms to explore and reveal potential long-range dependencies at the whole sequence level, aiming to capture distant interactions between binding sites. Benchmarking experiments demonstrate that ATP-Pred significantly improves the performance of ligand binding site prediction.

# 1. Requirements

Python >= 3.10.6

torch = 2.0.0

pandas = 2.0.0

scikit-learn = 1.2.2

ProtTrans (ProtT5-XL-UniRef50 model)

# 2. Datasets
Dataset1: taining set: ATP-227; test set: atp-17-for-227.txt
Dataset2: taining set: ATP-388; test set: atp-41-for-388.txt


# 3. How to use
## 3.1 Set up environment for ProtTrans
Set ProtTrans follow procedure from https://github.com/agemagician/ProtTrans/tree/master.

## 3.2 Extract features
Extract pLMs embedding: cd to the ATP-GLPred/feature_extract dictionary, and run "python3 extract_prot.py", the pLMs embedding matrixs will be extracted to Dataset/prot_embedding fold.
2、run python3 extract_prot.py to extract feature embedding.

## 3.3 Training and testing
cd to the ATP-GLPred, run python3 ATP-GLPred.py to train and test.
