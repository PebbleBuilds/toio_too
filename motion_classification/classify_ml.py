import csv, os, glob
import random
import math
import numpy as np
from numpy.fft import rfft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


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

# Make it explicit when we are truncating data
def dataTrunc(data, n):
    return data[:n]

# Zero-pad the data
def dataPad(data,max_len, n_features):
    rList = []
    diff = max_len - len(data)
    for i in range(diff):
        rList.append([[0.0] * n_features] * i + data + [[0.0] * n_features] * (diff - i))
    return rList

# Normalize (i.e. start at 0) the time for each dataset to avoid errors
def normalizeTime(data):
    n = data[0][0]
    for line in data:
        line[0] -= n
    return data

# Transform the data from each sample into something we can feed a classifier
def transformSample(data):
    lines = []
    for line in data:
        lines += line
    return lines

# Take the FFT of the variables and output them (up to truncated length)
# as one concatenated list
def rfftFeatures(data, numFeatures, truncLen):
    outlist = []
    # Implicitly skip timestamp feature
    for i in range(1,numFeatures):
        featfft = rfft(data[i::numFeatures])
        flat_trunc = []
        for i in range(truncLen):
            flat_trunc.append(featfft[i].real)
            flat_trunc.append(featfft[i].imag)
        outlist += flat_trunc
    return outlist

def main():
    categories = [loadData("./csv_data/circle"), loadData("./csv_data/halfmills"),loadData("./csv_data/holding"),
                    loadData("./csv_data/juggleparabola"), loadData("./csv_data/jugglesideways"),loadData("./csv_data/randomturn"),
                    loadData("./csv_data/shaking"),loadData("./csv_data/windmill")]
    # Split the data into tuples of processed data and labels
    dataPairs = []
    # The length to truncate each sample at (should find a more elegant solution for this eventually)
    #trunc_len = 199
    # Length to truncate fft at (based on trunc_len)
    #fft_len = 100
    # Number of features per sample
    n_features = 11
    # Use all possible pads if true, only one if false
    allPads = True
    #for i, c in enumerate(categories):
    #    for sample in c:
    #        x = transformSample(normalizeTime(dataTrunc(sample, trunc_len)))
    #        # Add FFT features
    #        x += rfftFeatures(transformSample(sample), n_features, fft_len)
    #        y = i
    #        dataPairs.append((x,y))
    max_len = 0
    min_len = math.inf
    for c in categories:
        for sample in c:
            max_len = max(max_len, len(sample))
            min_len = min(min_len, len(sample))
    fft_len = max_len // 2
    for i, c in enumerate(categories):
        for sample in c:
            #print(len(sample))
            #print(len(sample[0]))
            #exit()
            pads = dataPad(sample, max_len, n_features)
            for p in pads:
                x = transformSample(normalizeTime(p))
                # Add FFT features
                x += rfftFeatures(transformSample(sample), n_features, min_len // 2)
                y = i
                dataPairs.append((x,y))
                if not allPads:
                    break
    # Shuffle the pairs' order
    random.seed(0)
    np.random.seed(0)
    random.shuffle(dataPairs)
    tr_x = []
    tr_y = []
    test_x = []
    test_y = []
    #for x,y in dataPairs[:-20]:
    for x,y in dataPairs:
        tr_x.append(x)
        tr_y.append(y)
    #for x,y in dataPairs[-20:]:
    #    test_x.append(x)
    #    test_y.append(y)

    # Load test data
    test_categories = [loadData("./csv_testdata/circle"), loadData("./csv_testdata/halfmills"),loadData("./csv_testdata/holding"),
                loadData("./csv_testdata/juggleparabola"), loadData("./csv_testdata/jugglesideways"),loadData("./csv_testdata/randomturn"),
                loadData("./csv_testdata/shaking"),loadData("./csv_testdata/windmill")]

    for i, c in enumerate(test_categories):
        for sample in c:
            #print(len(sample))
            #print(len(sample[0]))
            #exit()
            pads = dataPad(sample, max_len, n_features)
            for p in pads:
                x = transformSample(normalizeTime(p))
                # Add FFT features
                x += rfftFeatures(transformSample(sample), n_features, min_len // 2)
                y = i
                test_x.append(x)
                test_y.append(y)
                if not allPads:
                    break
    #rf = KNeighborsClassifier()
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
        #print("Predicted:", p, "\tGround Truth:", y)
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
        #print("Predicted:", p, "\tGround Truth:", y)
        if p == y:
            val_corr += 1
        val_count += 1
    print()
    print("Validation Accuracy: ", (val_corr/val_count))

if __name__ == "__main__":
    main()
