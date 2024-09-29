import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import copy
import seaborn as sns
sns.set_style()

# CLASS 1.
class NNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential( nn.Linear(dim, 8),
                                 nn.Sigmoid(),
                                 nn.Linear(8, 4), 
                                 nn.Sigmoid(), 
                                 nn.Linear(4, 2),
                                 nn.Softmax()
        )
    def forward(self, Xtr):
        return self.fc(Xtr)

# CLASS 2.
class ModelWrapper():
    Xtr, Ytr, Xtest, Ytest = None, None, None, None 
    train_dataset, train_loader = None, None
    M, dim = 1024, 1 
    epochs, batch_size = 128, 32 
    criterion = None
    final_loss = None 
    @classmethod
    def generate_static_data(cls):
        if cls.Xtr == None:
            X = torch.tensor(np.random.normal(size=(cls.M, cls.dim)), dtype = torch.float32)
            Y = torch.tensor([[1,0] if x > 0 else [0,1] for x in X], dtype = torch.float32)
            cls.Xtr, cls.Xtest, cls.Ytr, cls.Ytest = train_test_split(X, Y, test_size = 0.3, random_state=1111)
            # IMP -> https://stackoverflow.com/questions/67683406/difference-between-dataset-and-tensordataset-in-pytorch
            cls.train_dataset = TensorDataset(ModelWrapper.Xtr, ModelWrapper.Ytr)
            cls.train_loader = DataLoader(cls.train_dataset, cls.batch_size, shuffle = False)
            cls.criterion = nn.CrossEntropyLoss()

    def __init__(self):
        np.random.seed(1919)
        self.generate_static_data()
        self.model = NNetwork(self.dim)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)

    '''
    Returns the list of weights for the trajectory.
    '''
    def trajectory_unroll(self):
        wts = []
        for epoch in range(ModelWrapper.epochs):
            wt, loss = self.trajectory_unroll_once()
            wts.append(wt)
        ModelWrapper.final_loss = loss 
        return wts
    '''
    Returns the weight after one iteration.
    '''
    def trajectory_unroll_once(self): 
        for xb, yb in ModelWrapper.train_loader:
            yp = self.model(xb)
            loss = ModelWrapper.criterion(yp, yb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        wt = []
        for idx, parameters in enumerate(self.model.parameters()):
            wt.append(copy.deepcopy(parameters.data))
        return wt, loss 

if __name__ == '__main__':
    mw = ModelWrapper()
    dat = mw.trajectory_unroll()
