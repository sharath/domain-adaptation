from common import init_experiment
from .utils import parse_args
from .baseline import baseline
from .gta import gta


args = parse_args()
if not args.baseline:
    experiment_name = init_experiment('GTA_adapt', args)
    gta(experiment_name, args)
else:
    experiment_name = init_experiment('GTA_baseline', args)
    baseline(experiment_name, args)