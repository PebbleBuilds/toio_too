import csv, os, glob
import random
import numpy as np
import numpy.random
from numpy.fft import rfft
import tensorflow as tf
import tensorflow.random
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


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
    trunc_len = 199
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
    np.random.seed(0)
    tf.random.set_seed(0)
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
    # Compose model
    tr_x = np.array(tr_x)
    tr_y = np.array(tr_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    model = Sequential([
        Flatten(input_shape=(len(tr_x[0]),)),
        #Dense(512, activation='relu',kernel_regularizer='l2'),
        #Dropout(0.3),
        Dense(256, activation='relu',kernel_regularizer='l2'),
        #Dropout(0.3),
        Dense(128, activation='relu',kernel_regularizer='l2'),
        #Dropout(0.3),
        Dense(64, activation='relu',kernel_regularizer='l2'),
        #Dropout(0.3),
        Dense(32, activation='relu',kernel_regularizer='l2'),
        #Dropout(0.2),
        Dense(len(categories))
    ])
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=loss_fn, metrics=['accuracy'])
    model.fit(tr_x, tr_y, epochs=5000)
    model.evaluate(test_x,  test_y, verbose=2)

if __name__ == "__main__":
    main()
