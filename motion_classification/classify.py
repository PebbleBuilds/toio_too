import csv, os, glob
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay


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

def main():
    randomLogs = loadData("./csv_data/random")
    windmillLogs = loadData("./csv_data/windmills")
    categories = [randomLogs, windmillLogs]
    # Split the data into tuples of processed data and labels
    dataPairs = []
    for i, c in enumerate(categories):
        for sample in c:
            x = transformSample(normalizeTime(dataTrunc(sample, 15)))
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
    rf = RandomForestClassifier()
    rf.fit(tr_x,tr_y)
    # Looking for overfitting as a sanity check
    print("Overfitting:")
    ps = rf.predict(tr_x)
    for p,y in zip(ps, tr_y):
        print("Predicted:", p, "\tGround Truth:", y)
    print()
    print("Validation:")
    ps = rf.predict(test_x)
    for p,y in zip(ps, test_y):
        print("Predicted:", p, "\tGround Truth:", y)

if __name__ == "__main__":
    main()
