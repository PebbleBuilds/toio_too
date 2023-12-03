import csv, os, glob
import random
import numpy as np


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

# Add FFT to numpy array
def addFFT(data_array, feature, window_size):
    #data_shape = list(data_array.shape)
    #data_shape[3] = 1
    #fft_array = np.zeros(data_shape)

    #fft_array = 