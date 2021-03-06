import os
import pickle
import torch
from time import time

def init_experiment(name, args, base_path='experiments', params_file='parameters.opt', log_file='output.log'):
    start = int(time()*100)
    experiment_name = f'{name}_{start}'
    experiment_dir = os.path.join(base_path, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    params_full_path = os.path.join(experiment_dir, params_file)
    with open(params_full_path, 'w') as fp:
        fp.write(str(args))
    log_path = os.path.join(experiment_dir, log_file)
    return experiment_name, log_path


def save_model(model, model_name, experiment_name, base_path='experiments'):
    import warnings
    warnings.filterwarnings('ignore')
    model = model.to('cpu')
    out_path = os.path.join(base_path, experiment_name, model_name + '.pt')
    torch.save(model, out_path)
    
    
def save_data(data, experiment_name, base_path='experiments'):
    out_path = os.path.join(base_path, experiment_name, 'results.dat')
    with open(out_path, 'wb') as fp:
        pickle.dump(data, fp)
    