import MODELS.model_param as g
import math 
import torch 
import numpy as np 
import json 
import copy as copy
# np.random.seed(1919)

class DataSet:
    def __init__(self) -> None:
        pass
    @staticmethod
    def getData():
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.float32)
        Y = torch.tensor([[0], [1], [1], [0]], dtype = torch.float32)
        return X, Y

class NN_SIGMOID:
    def __init__(self, lr = 0.001) -> None:
        self.X, self.Y = DataSet.getData()
        self.lr = lr
        ip_size = self.X.shape[-1]
        op_size0, op_size1 = 2, 1
        while True:
            self.layer0, self.bias0 = torch.tensor(np.random.random(size = (ip_size, op_size0)), dtype = torch.float32, requires_grad=True), \
                                    torch.tensor(np.random.random(size = (1, op_size0)), dtype = torch.float32, requires_grad=True)
            self.layer1, self.bias1 = torch.tensor(np.random.random(size = (op_size0, op_size1)), dtype = torch.float32, requires_grad=True), \
                                    torch.tensor(np.random.random(size = (1, op_size1)), dtype = torch.float32, requires_grad=True)
            with torch.no_grad():
                Yp = self.sigmoid(self.sigmoid((self.X @ self.layer0 + self.bias0)) @ self.layer1 + self.bias1)
                Loss = torch.mean((self.Y-Yp)**2)
                if g.loss_lb <= Loss and Loss <= g.loss_ub:
                    break                         

    def set_model_wts(self, model):
        with torch.no_grad():
            self.layer0 = copy.deepcopy(model.layer0) + torch.normal(0, g.std, size = model.layer0.shape)
            self.bias0 = copy.deepcopy(model.bias0) + torch.normal(0, g.std, size = model.bias0.shape)
            self.layer1 = copy.deepcopy(model.layer1) + torch.normal(0, g.std, size = model.layer1.shape)
            self.bias1 = copy.deepcopy(model.bias1) + torch.normal(0, g.std, size = model.bias1.shape)
        self.layer0.requires_grad, self.layer1.requires_grad, \
        self.bias0.requires_grad, self.bias1.requires_grad = True, True, True, True
    def sigmoid(self, x):
            return 1/(1 + torch.exp(-x))
    def once(self):
        Yp = self.sigmoid(self.sigmoid((self.X @ self.layer0 + self.bias0)) @ self.layer1 + self.bias1)
        Loss = torch.mean((self.Y-Yp)**2)
        Loss.backward()
        # UPDATE
        with torch.no_grad():
            self.layer0 -= self.lr * self.layer0.grad
            self.bias0 -= self.lr * self.bias0.grad 
            self.layer1 -= self.lr * self.layer1.grad 
            self.bias1 -= self.lr * self.bias1.grad 
        # Zero Grad 
        self.layer0.grad.zero_()
        self.bias0.grad.zero_()
        self.layer1.grad.zero_()
        self.bias1.grad.zero_()
        return Loss.item()
    def predict(self):
        with torch.no_grad():
            return self.sigmoid(self.sigmoid((self.X @ self.layer0 + self.bias0)) @ self.layer1 + self.bias1)
    def get_dist(self, model):
        with torch.no_grad():
            return math.sqrt(
                            torch.sum((self.layer0 - model.layer0)**2).item() + \
                            torch.sum((self.bias0 - model.bias0)**2).item() + \
                            torch.sum((self.layer1 - model.layer1)**2).item() + \
                            torch.sum((self.bias1 - model.bias1)**2).item() 
                            )
        

def run_model():
    obj_arr = [NN_SIGMOID(g.learning_rate) for _ in range(g.n_children)]
    for idx, obj in enumerate(obj_arr):
        if idx != 0: obj.set_model_wts(obj_arr[0])
    
    # UNIT TEST 1
    assert obj_arr[0].get_dist(obj_arr[0]) == 0, f'Wts should be equal.'
    # UNIT TEST 2
    for i in range(1, len(obj_arr)): 
        assert obj_arr[0].get_dist(obj_arr[1]) != 0, f'Wts should not be equal.'

    loss_arr = [[] for _ in range(g.n_children)] # N_CHILDREN * EPOCHS
    dist_arr = []
    
    for epoch in range(g.epochs):
        dist = 0
        for idx, obj in enumerate(obj_arr):
            loss_arr[idx].append(obj.once())
            dist += obj.get_dist(obj_arr[0])
        dist = dist / g.n_children 
        dist_arr.append(dist)
    with open('RAW_DATA/Sigmoid.json', 'w') as file:
        json.dump({
                    "avg_dist": dist_arr,
                    "loss": loss_arr
                 }, file)

if __name__ == '__main__':
    run_model()