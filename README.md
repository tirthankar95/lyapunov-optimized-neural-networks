# Description
Training large deep neural networks is resource intensive. This study investigates whether Lyapunov exponents can accelerate this process by aiding in the selection of hyperparameters. To study this I formulate an optimization problem using neural networks with different activation functions in the hidden layers. By initializing model weights with different random seeds, I calculate the Lyapunov exponent while performing traditional gradient descent on these model weights. The findings demonstrate that variations in the learning rate can induce chaotic changes in model weights. I also show that activation functions with more negative Lyapunov exponents exhibit better convergence properties. Additionally, the study also demonstrates that Lyapunov exponents can be utilized to select effective initial model weights for deep neural networks, improving the optimization process.

### Run code from main.ipynb
### Description of functions

`def lyapk_vs_loss(...)`

    '''
    Calculates if lower lyapunov gets me lower final loss.
    '''

    -> Runs class_nn.run_model()
        
        -> Takes a point, populates "avg distance" and "loss". 
        
        -> "avg distance" is distance to all neighbhouring points; averaged.
        
        -> "loss" -> [N_CHILDREN * EPOCHS], where N_CHILDREN is the number of neighbhouring points to the main point.

    -> Saves & Plots avg_lyapk & final loss.

---

`def plot_distance(...)`
    
    -> How distance evolves over epochs.

---

`def plot_lyapk(...)`

    -> How lyapunov exponents change over epochs.

---

`def plot_loss(...)`

    -> How loss of i-th point (parent/children) changes over epochs.

---

`def run(obj, filename, op_filename, lb, ub, n_points):`

    '''
    Calculates if learning rate can induce chaos.
    '''

    -> Sets learning rate from lb to ub in intervals of n_points.

    -> Runs class_nn.run_model().

    -> Saves result to op_filename, the Learning Rate and Lyapunov Exponent.
---

### Reference


Copyright (c) 2024 Tirthankar Mittra
