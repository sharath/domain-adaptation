import torch
from common import init_experiment
from .utils import parse_args
from .baseline import baseline
from .gta import gta

args = parse_args()
_ = torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if not args.baseline:
    experiment_name, log_file = init_experiment('GTA_adapt', args)
    with open(log_file, 'w') as fp:
        gta(experiment_name, args, fp)
else:
    experiment_name, log_file = init_experiment('GTA_baseline', args)
    with open(log_file, 'w') as fp:
        baseline(experiment_name, args, fp)