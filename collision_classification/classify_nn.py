from helpers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.signal import decimate

from torchvision import datasets
from torchvision.transforms import ToTensor

learning_rate = 1e-4
epochs = 10
batch_size = 5

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2608, 500)  # 3*3 from image dimension
        self.fc2 = nn.Linear(500, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
    
class Net2(nn.Module):

    def __init__(self, num_features, max_length, num_classes):
        super(Net2, self).__init__()
        # an affine operation: y = Wx + b
        self.fc0 = nn.Linear(num_features*max_length, 20)
        self.fc1 = nn.Linear(20, 84)
        #self.fc2 = nn.Linear(500,2608)
        #self.fc3 = nn.Linear(2608, 500)
        self.fc4 = nn.Linear(84, 20)
        self.fc5 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = F.dropout(x,p=0.2)
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    torch.manual_seed(0)
    categories = [loadData("./csv_data/november_11_collisions/hard_moving"),loadData("./csv_data/november_11_collisions/hard_still"),
                           loadData("./csv_data/november_11_collisions/soft_moving"),loadData("./csv_data/november_11_collisions/soft_still"),
                           loadData("./csv_data/november_11_collisions/no_collision")]
    
    # Split the data into tuples of processed data and labels
    dataPairs = []

    # Remove time and figure out max length
    max_length = 0
    num_samples = 0
    idx_to_keep = [1,2,3,5,6,4]
    for i, c in enumerate(categories):
        for sample_idx in range(0,len(c)):
            num_samples += 1
            c[sample_idx] = cropFeatures(c[sample_idx],idx_to_keep)
            if len(c[sample_idx]) > max_length:
                max_length = len(c[sample_idx])
    
    num_features = len(idx_to_keep)
    decimate_factor = 1
    max_length = max_length // decimate_factor
    X_all = np.zeros((num_samples, 1, max_length, num_features))
    y_all = np.zeros((num_samples))
    
    # Convert to np array, decimate, pad, and flatten
    sample_idx = 0
    for label, c in enumerate(categories):
        for arr in c:
            arr = np.asarray(arr)
            arr = decimate(arr,decimate_factor,axis=0)
            arr = np.pad(arr,[(0,max_length - arr.shape[0]),(0,0)],mode="reflect")
            

            # 5-class classification
            X_all[sample_idx,0] = arr
            y_all[sample_idx] = label 
            num_classes = 5

            # Classifying no collision vs collision
            #X_all[sample_idx,0] = arr
            #if label == 4:
            #   y_all[sample_idx] = 0
            #else:
            #   y_all[sample_idx] = 1
            #num_classes = 2

            # Classifying still vs moving vs no collision
            # X_all[sample_idx,0] = arr
            # if label == 0 or label == 2:
            #    y_all[sample_idx] = 0
            # elif label == 1 or label == 3:
            #    y_all[sample_idx] = 1
            # else:
            #    y_all[sample_idx] = 2
            # num_classes = 3

            sample_idx += 1

    # Normalize over time
    norms = np.linalg.norm(X_all,axis=2).reshape(num_samples,1,1,num_features)
    X_all = np.divide(X_all,norms)

    # Add FFT
    X_all = addFFT(X_all,0,20)
    X_all = addFFT(X_all,1,20)
    X_all = addFFT(X_all,2,20)
    X_all = addFFT(X_all,3,20)
    X_all = addFFT(X_all,4,20)
    num_features += 5

    # Stuff into dataloaders
    tensor_x = torch.Tensor(X_all).float() # transform to torch tensor
    tensor_y = torch.Tensor(y_all).long()
    my_dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(my_dataset,[0.8,0.2],generator)
    train_dataloader = DataLoader(train_set, batch_size=batch_size) # create your dataloader
    val_dataloader = DataLoader(val_set, batch_size=batch_size)

    # Create the net
    model = Net2(num_features, max_length, num_classes)

    # create your optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(val_dataloader, model, loss_fn)
    print("Done!")

if __name__ == "__main__":
    main()