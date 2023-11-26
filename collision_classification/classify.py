import csv, os, glob
import random
import numpy as np
from numpy.fft import rfft
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.svm import SVC

from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

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
def multi_feature_DTW(a, b, num_features=11):
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

    # Preprocess everything into np arrays
    max_length = 0
    num_samples = 0
    for i, c in enumerate(categories):
        for sample in c:
            num_samples += 1
            sample = removeTimeFeature(sample)
            if len(sample) > max_length:
                max_length = len(sample)
    
    num_features = 11
    X_train = np.zeros((num_samples, max_length * num_features))
    y_train = np.zeros((num_samples))

    sample_idx = 0
    for label, c in enumerate(categories):
        for sample in c:
            arr = np.asarray(sample)
            arr = np.pad(arr,[(0,max_length - len(sample)),(0,0)],mode="constant")
            arr = arr.flatten()
            X_train[sample_idx] = arr
            y_train[sample_idx] = label
            sample_idx += 1
            
    parameters = {'n_neighbors':[1]}
    knn = KNeighborsClassifier(metric=multi_feature_DTW)
    clf = GridSearchCV(knn, parameters, cv=2, verbose=1)
    clf.fit(X_train, y_train)



    """
    # Length to truncate fft at (based on trunc_len)
    fft_len = 100
    # Number of features per sample
    n_features = 11
    for i, c in enumerate(categories):
        for sample in c:
            x = transformSample(normalizeTime(dataTrunc(sample, trunc_len)))
            # Add FFT features
            x += rfftFeatures(transformSample(sample), n_features, fft_len)
            y = i
            dataPairs.append((x,y))
            
    # Shuffle the pairs' order
    random.seed(0)
    random.shuffle(dataPairs)
    tr_x = []
    tr_y = []
    test_x = []
    test_y = []
    for x,y in dataPairs[:-2]:
        tr_x.append(x)
        tr_y.append(y)
    for x,y in dataPairs[-2:]:
        test_x.append(x)
        test_y.append(y)
    #rf = RandomForestClassifier()
    rf = SVC()
    rf.fit(tr_x,tr_y)
    # Looking for overfitting as a sanity check
    print("Overfitting:")
    ps = rf.predict(tr_x)
    # Count correct and total predictions to get ratio
    tr_corr = 0
    tr_count = 0
    for p,y in zip(ps, tr_y):
        print("Predicted:", p, "\tGround Truth:", y)
        if p == y:
            tr_corr += 1
        tr_count += 1
    print()
    print("Training Accuracy: ", (tr_corr/tr_count))
    print()
    val_corr = 0
    val_count = 0
    print("Validation:")
    ps = rf.predict(test_x)
    for p,y in zip(ps, test_y):
        print("Predicted:", p, "\tGround Truth:", y)
        if p == y:
            val_corr += 1
        val_count += 1
    print()
    print("Validation Accuracy: ", (val_corr/val_count))
    """

if __name__ == "__main__":
    main()
