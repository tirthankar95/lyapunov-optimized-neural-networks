from kantz_nn import * 
# import multiprocessing as mpc
from GLOBAL import global_module as g
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm
sns.set_style("whitegrid")

if __name__ == '__main__':
    
    # # First Model.
    print(f'Model 1...')
    from NNModel import model1 as m1 
    shm_list =  []
    for _ in tqdm(range(g.parallel_points)):
        shm_list.append(run_nn(m1.ModelWrapper))
    X, Y = [x[0] for x in shm_list], [x[1] for x in shm_list]
    plt.figure()
    sns.regplot(x = X, y = Y, color='magenta')
    plt.xlabel('Lyapunov Exponent')
    plt.ylabel('Loss')
    plt.savefig('model1_chaos_optim.png')
    # plt.show()
    print(f'Model 1: LyapK[{sum(X)/len(X)}], Loss[{sum(Y)/len(Y)}]')
    
    # # Second Model.
    print(f'Model 2...')
    from NNModel import model1_sigmoid as m2 
    shm_list =  []
    for _ in tqdm(range(g.parallel_points)):
        shm_list.append(run_nn(m2.ModelWrapper))
    X, Y = [x[0] for x in shm_list], [x[1] for x in shm_list]
    plt.figure()
    sns.regplot(x = X, y = Y, color='magenta')
    plt.xlabel('Lyapunov Exponent')
    plt.ylabel('Loss')
    plt.savefig('model2_chaos_optim.png')
    # plt.show()
    print(f'Model 2: LyapK[{sum(X)/len(X)}], Loss[{sum(Y)/len(Y)}]')

    # Thrid Model.
    print(f'Model 3...')
    from NNModel import model1_tanh as m3 
    shm_list =  []
    for _ in tqdm(range(g.parallel_points)):
        shm_list.append(run_nn(m3.ModelWrapper))
    X, Y = [x[0] for x in shm_list], [x[1] for x in shm_list]
    plt.figure()
    sns.regplot(x = X, y = Y, color='magenta')
    plt.xlabel('Lyapunov Exponent')
    plt.ylabel('Loss')
    plt.savefig('model3_chaos_optim.png')
    # plt.show()
    print(f'Model 3: LyapK[{sum(X)/len(X)}], Loss[{sum(Y)/len(Y)}]')

    # Fourth Model.
    print(f'Model 4...')
    from NNModel import model1_tanh as m3 
    shm_list =  []
    for _ in tqdm(range(g.parallel_points)):
        shm_list.append(run_nn(m3.ModelWrapper))
    X, Y = [x[0] for x in shm_list], [x[1] for x in shm_list]
    plt.figure()
    sns.regplot(x = X, y = Y, color='blue')
    plt.xlabel('Lyapunov Exponent')
    plt.ylabel('Loss')
    plt.savefig('model4_chaos_optim.png')
    # plt.show()
    print(f'Model 4: LyapK[{sum(X)/len(X)}], Loss[{sum(Y)/len(Y)}]')