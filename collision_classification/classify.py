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
def multi_feature_DTW(a, b, num_features):   
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
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
        new_data = data[1:]
    return new_data

def main():
    categories = [loadData("./csv_data/november_11_collisions/hard_moving"),loadData("./csv_data/november_11_collisions/hard_still"),
                           loadData("./csv_data/november_11_collisions/soft_moving"),loadData("./csv_data/november_11_collisions/soft_still"),
                           loadData("./csv_data/november_11_collisions/no_collision")]
    # Split the data into tuples of processed data and labels
    dataPairs = []

    # Remove time feature of every ts and find the longest one
    max_length = 0
    for i, c in enumerate(categories):
        for sample in c:
            sample = removeTimeFeature(sample)
            if len(sample) > max_length:
                max_length = len(sample)

    # constant pad every sample
    print(max_length)
    for i, c in enumerate(categories):
        for sample in c:
            print(len(sample))
            sample = np.pad(sample,max_length - len(sample),mode="constant")

    # stuff everything in an np array



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
