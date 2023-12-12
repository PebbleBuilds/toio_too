import csv, os, glob
import random
import numpy as np
from numpy.fft import rfft
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.svm import SVC

from scipy.spatial import distance
from scipy.signal import decimate
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import decomposition
import time

from helpers import *

import pandas as pd

def main():
    categories = [loadData("./csv_data/november_11_collisions/hard_moving"),loadData("./csv_data/november_11_collisions/hard_still"),
                           loadData("./csv_data/november_11_collisions/soft_moving"),loadData("./csv_data/november_11_collisions/soft_still")]#,
                           #loadData("./csv_data/november_11_collisions/no_collision")]
    
    # Split the data into tuples of processed data and labels
    dataPairs = []

    # Remove time and figure out max length
    max_length = 0
    num_samples = 0
    idx_to_keep = [1,2]

    for i, c in enumerate(categories):
        for sample_idx in range(0,len(c)):
            num_samples += 1
            c[sample_idx] = cropFeatures(c[sample_idx],idx_to_keep)
            if len(c[sample_idx]) > max_length:
                max_length = len(c[sample_idx])
    
    num_features = len(idx_to_keep)
    X_all = np.zeros((num_samples, max_length * num_features))
    y_all = np.zeros((num_samples))

    print(max_length)
    
    # Convert to np array, decimate, pad, normalize, and flatten
    sample_idx = 0
    for i, c in enumerate(categories):
        for arr in c:
            # Classifying moving vs still (use idx 5 and 6)
            # if i == 0 or i == 2:
            #     y_all[sample_idx] = 0
            # else:
            #     y_all[sample_idx] = 1
            # arr = np.asarray(arr)
            # arr = np.pad(arr,[(0,max_length - arr.shape[0]),(0,0)],mode="edge")
            # arr = addGradient_sk(arr,0)
            # arr = addGradient_sk(arr,1)
            # arr = arr[:,2:].flatten()
            # X_all[sample_idx] = arr

            # 5-fold classification
            #y_all[sample_idx] = i

            # Classifying soft vs hard
            if i == 0 or i == 1:
                y_all[sample_idx] = 0
            else:
                y_all[sample_idx] = 1
            arr = np.asarray(arr)
            arr = np.pad(arr,[(0,max_length - arr.shape[0]),(0,0)],mode="edge")
            arr = addCepstral_sk(arr,0,110)
            arr = addCepstral_sk(arr,1,110)
            #arr = addCepstral_sk(arr,2,110)
            arr = arr[:,num_features:].flatten()
            X_all[sample_idx] = arr

            sample_idx += 1

    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

    np.savetxt("foo.csv", X_train, delimiter=",")

    #df = pd.DataFrame (X_train)
    #filepath = 'my_excel_file.xlsx'
    #df.to_excel(filepath, index=False)
            
    svc = SVC(C=10,gamma="scale",kernel="rbf",tol=1e-5)
    svc.fit(X_train, y_train)

    """
    print("Predicting train samples:")
    start = time.time()
    y_pred = svc.predict(X_train)
    print(classification_report(y_train, y_pred))
    print("Prediction completed in this many seconds:",time.time() - start)
    """

    print("Predicting validation samples:")
    start = time.time()
    y_pred = svc.predict(X_val)
    print(classification_report(y_val, y_pred))
    print("Prediction completed in this many seconds:",time.time() - start)
    print(y_val)
    print(y_pred)

if __name__ == "__main__":
    main()
