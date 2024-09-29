import MODELS.model_param as g
import json 
import math


def get_lyapk(filename) -> int:
    def load_param_from_file(filename, param):
        var = None 
        with open(f"{filename}", 'r') as file:
            var = json.load(file)
            return var[f"{param}"]
    dist = load_param_from_file(f"{filename}", "avg_dist")
    l_sum, l_cnt = 0, 0
    for i in range(1, len(dist)):
        if dist[i] > 1e9: break 
        l_sum += math.log(abs(dist[i]/dist[i-1]))
        l_cnt += 1 
    return l_sum/l_cnt

def run(obj, filename, op_filename, lb, ub, n_points):
    global save_op_file
    delta = (ub - lb)/n_points 
    lyapk_arr, lr_arr = [], []
    while lb < ub:
        g.set_lr(lb)
        obj.run_model()
        lr_arr.append(lb)
        lyapk_arr.append(get_lyapk(filename))
        lb += delta
    with open(f"{op_filename}", 'w') as file:
        json.dump({
                    "x": lr_arr,
                    "y": lyapk_arr
                }, file)
