import csv, os, glob
import random
import numpy as np
import scipy.fftpack as fft


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

# Get rid of time as a feature - everything is already a uniformly sampled time series.
def removeTimeFeature(data):
    new_data = []
    for row in data:
        new_data.append(row[1:])
    return new_data

# Get rid of other features
def cropFeatures(data,idx_to_keep):
    new_data = []
    for row in data:
        new_row = []
        for i in idx_to_keep:
            new_row.append(row[i])
        new_data.append(new_row)
    return new_data

# normalize features right before padding
def normalizeFeatures(arr):
    a = (arr-np.min(arr,axis=0))
    b = (np.max(arr,axis=0)-np.min(arr,axis=0))
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
# Add FFT to numpy array. For sklearn.
def addFFT_sk(data_array, feature, window_size):
    data_shape = list(data_array.shape)
    data_shape[1] += 1
    new_data_array = np.zeros(data_shape)
    if data_shape[0] % window_size != 0:
        print("Window size of %d does not divide time length of %d",(window_size,data_shape[1]))
        assert False
    new_data_array[0:data_array.shape[0],0:data_array.shape[1]] = data_array

    for i in range(0,data_shape[0],window_size):
        new_data_array[i:i+window_size,-1] = np.abs(fft.rfft(data_array[i:i+window_size,feature],axis=0))
    return new_data_array

def addGradient_sk(data_array, feature):
    data_shape = list(data_array.shape)
    data_shape[1] += 1
    new_data_array = np.zeros(data_shape)
    new_data_array[0:data_array.shape[0],0:data_array.shape[1]] = data_array
    new_data_array[:,-1] = np.gradient(data_array[:,feature])
    return new_data_array