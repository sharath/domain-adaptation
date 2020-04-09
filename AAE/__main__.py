from common import init_experiment
from .utils import parse_args
from .baseline import baseline
from .aae import aae


args = parse_args()
if not args.baseline:
    experiment_name, log_file = init_experiment('AAE_adapt', args)
    with open(log_file, 'w') as fp:
        aae(experiment_name, args, fp)
else:
    experiment_name, log_file = init_experiment('AAE_baseline', args)
    with open(log_file, 'w') as fp:
        baseline(experiment_name, args, fp)