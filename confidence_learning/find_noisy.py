import argparse
import numpy as np
import pandas as pd
import os
import sys
import time
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

import cleanlab
from cleanlab.pruning import get_noise_indices

model = 'clean_embed_all-mpnet-base-v2.csv'
df = pd.read_csv('/global/project/hpcg1614_shared/ca/data/banking77/{}'.format(model))
df_orig = pd.read_csv('clean.csv')

#df = df.head(1000)
#df_orig = df_orig.head(1000)

df_orig = df_orig.to_numpy()

from sklearn.model_selection import train_test_split, StratifiedKFold

X = df.drop(['category'], axis=1).to_numpy()
y_cat = df['category'].to_numpy()
label_transformer = LabelEncoder()
y = label_transformer.fit_transform(y_cat)


kfold = StratifiedKFold(n_splits=10, shuffle=True)

res = []

split = 0
for train_ix, val_ix in kfold.split(X, y):
    split = split + 1
    print("split")
    X_train, X_val = X[train_ix], X[val_ix]  
    y_train, y_val = y[train_ix], y[val_ix] 
    X_orig_val = df_orig[val_ix]

    params = {
              "learning_rate": 0.1,
              "max_depth": 4,
              "num_leaves": 15,
              "n_estimators": 1000,
              "n_jobs": 5,
              "verbosity": -1,
              "seed": 77,
        }
    estimator = LGBMClassifier(**params)
    estimator.fit(X_train, y_train)

    y_val_pred =estimator.predict_proba(X_val)

    ordered_label_errors = get_noise_indices(
        s=y_val,
        psx=y_val_pred,
        sorted_index_method='normalized_margin', # Orders label errors
    )
   
    i = 0
    for error_ix in ordered_label_errors:
        i = i + 1
        print()
        print("Possible Truth Label Error:".format(error_ix))
        
        o_ix = val_ix[error_ix]
        print("  Orig IDX: {}".format(o_ix))
        print("  Orig Message: {}".format(df_orig[o_ix]))
        print("  Message:      {}".format(X_orig_val[error_ix][0]))
        print("  Truth Label:  {} {}".format(y_val[error_ix], label_transformer.inverse_transform([y_val[error_ix]])))
        
        
        probas = np.around(y_val_pred[error_ix], decimals=4)
        index_max = np.argmax(probas)
        print("  Predicted label: {}".format(label_transformer.inverse_transform([index_max])))
        print("  Predicted probas: {}".format(probas))
        
        res.append({
            'Split': split,
            'i': i,
            'ID': o_ix,
            'Message': X_orig_val[error_ix][0],
            'Truth': label_transformer.inverse_transform([y_val[error_ix]])[0],
            'Maybe Better': label_transformer.inverse_transform([index_max])[0],
        })
        
df_res = pd.DataFrame(res)
df_res.to_csv('possible_errors.csv', index=False)
        
