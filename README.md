# banking77
Exploring possible errors on BANKING77

## data folder
raw BANKING77 training data, EDA on raw, and 2 versions of corrected data: 1) removing all potentially incorrect labels, 2) updated based on alternative labels from find_errors

## find_errors folder
2 ways to identify potential label errors: 1) Confidence Learning (CL) Framework and 2) Cosine-similarity approach
Outputs from both aprroaches in \out

## experiments
experiments with corrected data in supervised (LGBM) and unsupervised (agglomerative clustering) intent classifier
 