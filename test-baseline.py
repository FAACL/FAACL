import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import random
import tensorflow as tf
from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg
from flearn.trainer.fesem import FeSEM
from flearn.trainer.fedgroup import FedGroup
from flearn.trainer.ifca import IFCA

def main(dataset, model, trainer, glr, dynamic, swap_p, shift_type, RAC, RCC, group = 1, cluster = 2, seed = 2077):
    config = TrainConfig(dataset, model, trainer, group =group, swap_label = swap_label, seed = seed)
    config.trainer_config['group_agg_lr'] = glr
    config.trainer_config['RAC'] = RAC
    config.trainer_config['RCC'] = RCC
    config.trainer_config['dynamic'] = dynamic
    config.trainer_config['swap_p'] = swap_p
    config.trainer_config['shift_type'] = shift_type
    config.trainer_config['num_group'] = cluster
    if group == 1:
        config.results_path += "iid/"
    elif group != 1:
        config.results_path += "group_" + str(group) + "/"
    
    config.results_path += "seed_" + str(seed) + "/"

    if trainer == 'fedavg':
        trainer = FedAvg(config)
    if trainer == 'fesem':
        trainer = FeSEM(config)
    if trainer == 'ifca':
        trainer = IFCA(config)
    if trainer == 'fedgroup':
        trainer = FedGroup(config)
    trainer.train()
    

Dataset = ['mnist', 'emnist', 'fmnist']
Model = ['mlp']
Trainer = ['ifca', 'fedavg', 'fesem', 'fedgroup']
Group = [1,2,3,6,7,8] 
Seed = [10, 55, 2077]

for dataset in Dataset:
    for model in Model:
        for trainer in Trainer:
            for group in Group:
                for seed in Seed:
                    main(dataset=dataset,model=model,trainer=trainer,glr=0,dynamic=False,swap_p=0, \
        shift_type=None, RAC=False, RCC=False,group=group, cluster = 5, seed=seed)


dataset = 'femnist'
model = 'mlp'
Trainer = ['ifca', 'fedavg', 'fesem', 'fedgroup']
group = 0
Seed = [10, 55, 2077]

for trainer in Trainer:
    for seed in Seed:
        main(dataset=dataset,model=model,trainer=trainer,glr=0,dynamic=False,swap_p=0, \
shift_type=None, RAC=False, RCC=False,group=group, cluster = 5, seed=seed)