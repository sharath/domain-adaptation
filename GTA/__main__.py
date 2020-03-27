from common import init_experiment
from .utils import parse_args
from .baseline import baseline
from .gta import gta

args = parse_args()
experiment_name, log_file = init_experiment(args)

with open(log_file, 'w') as fp:
    if not args.baseline:
        gta(experiment_name, args, fp)
    else:
        baseline(experiment_name, args, fp)