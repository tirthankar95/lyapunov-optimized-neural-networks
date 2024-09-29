import torch
import torch.nn as nn 
import torch.optim as optim
import multiprocessing as mpc 
from GLOBAL import global_module as g 
import math
from typing import List 
import numpy as np 

dist, child_models = None, None
# def load_parameters(model, fName):
#     with torch.no_grad():
#         for idx, layer in enumerate(model.parameters()):
#             layer = torch.load(f'dat/{fName}_layer{idx}')

# def save_parameters(model, fName):
#     with torch.no_grad():
#         for idx, layer in enumerate(model.parameters()):
#             torch.save(layer, f'dat/{fName}_layer{idx}')

def sanity_check_wts(m1, m2):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1,p2)
'''
Returns the list of distance measures.
'''
def get_distance(indx, cModel, p_wts) -> List[int]:
    global dist
    for _ in range(g.kantz_iter):
        cModel.trajectory_unroll_once()
        p_wt = p_wts[_]
        _sum_ = 0 
        for idx, param in enumerate(cModel.model.parameters()):
            _sum_ += torch.sum((param.data - p_wt[idx])**2).item()
        temp_dist = math.sqrt(_sum_)
        dist[indx][_] = temp_dist
'''
Test:
for param1, param2 in zip(obj1.parameters(), obj.parameters()):
    assert torch.equal(param1, param2)
'''
def init_wt(nn_class, pModel):
    global child_models
    cModel = nn_class()
    cModel.model.load_state_dict(pModel.state_dict())
    for idx, lc in enumerate(cModel.model.parameters()):
        lc.data = lc.data + torch.normal(0, g.std, size = lc.shape)
    child_models.append(cModel)

def run_nn(nn_class):
    global dist, child_models
    parent_model = nn_class()
    g.kantz_iter = min(nn_class.epochs, g.kantz_iter)
    dist, child_models = [[_ for _ in range(g.kantz_iter)] for __ in range(g.child_n) ], []
    # COPYING SIMILAR WTS TO CHILD FROM PARENT.
    for _ in range(g.child_n):
        init_wt(nn_class, parent_model.model)
    # TRAJECTORY UNROLL. 
    p_wts = parent_model.trajectory_unroll()
    for _ in range(g.child_n):
        get_distance(_, child_models[_], p_wts)
    # CALCULATE LYAPK
    lyapk, lyapk_arr = 0, []
    for t in range(g.kantz_iter):
        temp_dist = 0
        for _ in range(g.child_n):
            temp_dist += dist[_][t]
        temp_dist /= g.child_n
        lyapk_arr.append(temp_dist)
    sz = len(lyapk_arr) - 1
    lyapk_temp = [ np.log(abs(lyapk_arr[i+1]/lyapk_arr[i])) for i in range(sz)]
    return np.mean(lyapk_temp), nn_class.final_loss.item()
