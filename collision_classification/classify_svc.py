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
                           loadData("./csv_data/november_11_collisions/soft_moving"),loadData("./csv_data/november_11_collisions/soft_still"),
                           loadData("./csv_data/november_11_collisions/no_collision")]
    
    # Split the data into tuples of processed data and labels
    dataPairs = []

    # Remove time and figure out max length
    max_length = 0
    num_samples = 0
    idx_to_keep = [1,2,3,4,5,6,7,8]

    for i, c in enumerate(categories):
        for sample_idx in range(0,len(c)):
            num_samples += 1
            c[sample_idx] = cropFeatures(c[sample_idx],idx_to_keep)
            if len(c[sample_idx]) > max_length:
                max_length = len(c[sample_idx])
    
    num_features = len(idx_to_keep)
    decimate_factor = 1
    max_length = max_length // decimate_factor
    X_all = np.zeros((num_samples, max_length * num_features))
    y_all = np.zeros((num_samples))

    print(max_length)
    
    # Convert to np array, decimate, pad, normalize, and flatten
    sample_idx = 0
    for label, c in enumerate(categories):
        for arr in c:
            arr = np.asarray(arr)
            arr = decimate(arr,decimate_factor,axis=0)
            arr = np.pad(arr,[(0,max_length - arr.shape[0]),(0,0)],mode="wrap")
            #arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
            arr = arr.flatten()
            X_all[sample_idx] = arr
            y_all[sample_idx] = label
            sample_idx += 1

    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    np.savetxt("foo.csv", X_train, delimiter=",")

    #df = pd.DataFrame (X_train)
    #filepath = 'my_excel_file.xlsx'
    #df.to_excel(filepath, index=False)
            
    svc = SVC(C=1,gamma="auto")
    svc.fit(X_train, y_train)

    print("Predicting...")
    start = time.time()
    y_pred = svc.predict(X_val)
    print(classification_report(y_val, y_pred))
    print("Prediction completed in this many seconds:",time.time() - start)
    print(y_val)
    print(y_pred)

if __name__ == "__main__":
    main()