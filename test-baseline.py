#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import random
import tensorflow as tf
from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg
from flearn.trainer.fesem import FeSEM
from flearn.trainer.fedgroup import FedGroup
from flearn.trainer.ifca import IFCA
from flearn.trainer.fedsoft import FedSoft

def main(dataset, model, trainer, glr, dynamic, swap_p, shift_type, RAC, RCC, group = 1, swap_label = 0, cluster = 2, seed = 2077):
    config = TrainConfig(dataset, model, trainer, group =group, swap_label = swap_label, seed = seed)
    config.trainer_config['group_agg_lr'] = glr
    config.trainer_config['RAC'] = RAC
    config.trainer_config['RCC'] = RCC
    config.trainer_config['dynamic'] = dynamic
    config.trainer_config['swap_p'] = swap_p
    config.trainer_config['shift_type'] = shift_type
    config.trainer_config['num_group'] = cluster
    config.results_path = '../results/'+ dataset+ "/"+ trainer +"/"
    
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
    if trainer == 'fedsoft':
        trainer = FedSoft(config)
    trainer.train()
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--trainer')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--group', type=int)
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--cluster', type = int, default=2)
    args = parser.parse_args()
    print(args)
    main(dataset=args.dataset,model=args.model,trainer=args.trainer,
         glr=0,dynamic=False,swap_p=0, shift_type=None, RAC=False, 
         RCC=False,group=args.group, swap_label = 0, seed=args.seed, cluster=args.cluster)
