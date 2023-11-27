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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import time

# Load all csv file data from selected directory
def loadData(path):
    #path = './csv_data'
    # Note current working directory for later
    cwd = os.getcwd()
    # Find all csv files in specified path
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    data = []
    for f in result:
        rows = []
        with open(f, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            # Eliminate category names
            csvreader = [x for x in csvreader][:]
            for row in csvreader[1:]:
                # Remove annoying formatting
                row = str(row)[1:-1].replace("'", '').split(",")
                # Convert non-empty fields to float
                row = [float(x) for x in row if x != '']
                rows.append(row)
        data.append(rows)
    # Go back to original working directory before leaving
    os.chdir(cwd)
    return data

# https://stackoverflow.com/questions/57015499/how-to-use-dynamic-time-warping-with-knn-in-python
def multi_feature_DTW(a, b, num_features=10):
    a_reshaped = a.reshape(a.size//num_features,num_features)
    b_reshaped = b.reshape(b.size//num_features,num_features)

    an = a_reshaped.shape[0]
    bn = b_reshaped.shape[0]

    pointwise_distance = distance.cdist(a_reshaped,b_reshaped)
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

    return cumdist[an, bn]

# Get rid of time as a feature - everything is already a uniformly sampled time series.
def removeTimeFeature(data):
    new_data = []
    for row in data:
        new_data.append(row[1:])
    return new_data

def main():
    categories = [loadData("./csv_data/november_11_collisions/hard_moving"),loadData("./csv_data/november_11_collisions/hard_still"),
                           loadData("./csv_data/november_11_collisions/soft_moving"),loadData("./csv_data/november_11_collisions/soft_still"),
                           loadData("./csv_data/november_11_collisions/no_collision")]
    
    # Split the data into tuples of processed data and labels
    dataPairs = []

    # Remove time and figure out max length
    max_length = 0
    num_samples = 0
    for i, c in enumerate(categories):
        for sample in c:
            num_samples += 1
            sample = removeTimeFeature(sample)
            if len(sample) > max_length:
                max_length = len(sample)
    
    num_features = 10
    decimate_factor = 10
    max_length = max_length // decimate_factor
    X_all = np.zeros((num_samples, max_length * num_features))
    y_all = np.zeros((num_samples))
    
    # Convert to np array, decimate, pad, and flatten
    sample_idx = 0
    for label, c in enumerate(categories):
        for arr in c:
            arr = np.asarray(sample)
            arr = decimate(arr,5,axis=0)
            arr = np.pad(arr,[(0,max_length - arr.shape[0]),(0,0)],mode="wrap")
            arr = arr.flatten()
            X_all[sample_idx] = arr
            y_all[sample_idx] = label
            sample_idx += 1

    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.66, random_state=42)
            
    knn = KNeighborsClassifier(metric=multi_feature_DTW)
    knn.fit(X_train, y_train)

    print("Predicting...")
    start = time.time()
    y_pred = knn.predict(X_val)
    print(classification_report(y_val, y_pred))
    print("Prediction completed in this many seconds:",time.time() - start)
    print(y_val)
    print(y_pred)

if __name__ == "__main__":
    main()
