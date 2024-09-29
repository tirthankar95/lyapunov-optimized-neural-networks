n_children = 25
epochs = 10
learning_rate = 0.01
std = 0.0001
loss_lb, loss_ub = 0.2, 0.4

def set_lr(val: int):
    global learning_rate
    learning_rate = val