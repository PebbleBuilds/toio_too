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
    # Test 1: Nov 11 train, Dec 10 test
    # categories_train = [loadData("./csv_data/november_11_collisions/hard_moving"),loadData("./csv_data/november_11_collisions/hard_still"),
    #                        loadData("./csv_data/november_11_collisions/soft_moving"),loadData("./csv_data/november_11_collisions/soft_still")]
    
    # categories_test = [loadData("./csv_data/dec_10_collisions/hard_moving"),loadData("./csv_data/dec_10_collisions/hard_still"),
    #                        loadData("./csv_data/dec_10_collisions/soft_moving"),loadData("./csv_data/dec_10_collisions/soft_still")]
    
    # Test 2: Nov 11 + Dec 10 train, Dec 13 test
    categories_train = [loadData("./csv_data/november_11_collisions/hard_moving")+loadData("./csv_data/dec_10_collisions/hard_moving"),
                        loadData("./csv_data/november_11_collisions/hard_still") +loadData("./csv_data/dec_10_collisions/hard_still"),
                        loadData("./csv_data/november_11_collisions/soft_moving")+loadData("./csv_data/dec_10_collisions/soft_moving"),
                        loadData("./csv_data/november_11_collisions/soft_still")+loadData("./csv_data/dec_10_collisions/soft_still")]
    
    categories_test = [loadData("./csv_data/dec_13_collisions/hard_moving"),loadData("./csv_data/dec_13_collisions/hard_still"),
                           loadData("./csv_data/dec_13_collisions/soft_moving"),loadData("./csv_data/dec_13_collisions/soft_still")]

    # Crop features. Max length will be 660.
    idx_to_keep = [1,2]
    max_length = 660

    num_samples_train = 0
    for i, c in enumerate(categories_train):
        for sample_idx in range(0,len(c)):
            num_samples_train += 1
            c[sample_idx] = cropFeatures(c[sample_idx],idx_to_keep)

    num_samples_test = 0
    for i, c in enumerate(categories_test):
        for sample_idx in range(0,len(c)):
            num_samples_test += 1
            c[sample_idx] = cropFeatures(c[sample_idx],idx_to_keep)
    
    # Pre-allocate data arrays
    num_features = len(idx_to_keep) 
    X_train = np.zeros((num_samples_train, max_length * num_features))
    y_train = np.zeros((num_samples_train))
    X_test = np.zeros((num_samples_test, max_length * num_features))
    y_test = np.zeros((num_samples_test))
    
    # Convert to np array, decimate, pad, normalize, and flatten
    sample_idx = 0
    for i, c in enumerate(categories_train):
        for arr in c:
            # Classifying moving vs still (use idx 5 and 6)
            # if i == 0 or i == 2:
            #     y_train[sample_idx] = 0
            # else:
            #     y_train[sample_idx] = 1
            # arr = np.asarray(arr)
            # arr = np.pad(arr,[(0,max_length - arr.shape[0]),(0,0)],mode="edge")
            # arr = addGradient_sk(arr,0)
            # arr = addGradient_sk(arr,1)
            # arr = arr[:,2:].flatten()
            # X_train[sample_idx] = arr

            # Classifying soft vs hard (use idx 1 and 2)
            if i == 0 or i == 1:
                y_train[sample_idx] = 0
            else:
                y_train[sample_idx] = 1
            arr = np.asarray(arr)
            arr = np.pad(arr,[(0,max_length - arr.shape[0]),(0,0)],mode="edge")
            arr = addCepstral_sk(arr,0,110)
            arr = addCepstral_sk(arr,1,110)
            arr = arr[:,num_features:].flatten()
            X_train[sample_idx] = arr

            sample_idx += 1

    # Convert to np array, decimate, pad, normalize, and flatten
    sample_idx = 0
    for i, c in enumerate(categories_test):
        for arr in c:
            # Classifying moving vs still (use idx 5 and 6)
            # if i == 0 or i == 2:
            #     y_test[sample_idx] = 0
            # else:
            #     y_test[sample_idx] = 1
            # arr = np.asarray(arr)
            # arr = np.pad(arr,[(0,max_length - arr.shape[0]),(0,0)],mode="edge")
            # arr = addGradient_sk(arr,0)
            # arr = addGradient_sk(arr,1)
            # arr = arr[:,2:].flatten()
            # X_test[sample_idx] = arr

            # Classifying soft vs hard
            if i == 0 or i == 1:
                y_test[sample_idx] = 0
            else:
                y_test[sample_idx] = 1
            arr = np.asarray(arr)
            arr = np.pad(arr,[(0,max_length - arr.shape[0]),(0,0)],mode="edge")
            arr = addCepstral_sk(arr,0,110)
            arr = addCepstral_sk(arr,1,110)
            arr = arr[:,num_features:].flatten()
            X_test[sample_idx] = arr

            sample_idx += 1
            
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
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Prediction completed in this many seconds:",time.time() - start)
    print(y_test)
    print(y_pred)

if __name__ == "__main__":
    main()
