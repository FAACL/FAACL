#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import collections
import importlib
import tensorflow as tf
import random
import time
from termcolor import colored
import copy
from utils.read_data import read_federated_data
from utils.trainer_utils import TrainConfig
from utils.trainer_utils import process_grad, calculate_cosine_dissimilarity
from flearn.server import Server
from flearn.client import Client
from flearn.group import Group
from collections import Counter
from utils.export_result import ResultWriter
from math import ceil
import math
import os
import tensorflow as tf
from tensorflow import keras
import scipy.stats as stats

class Centralize(object):
    def __init__(self, train_config):
        for key, val in train_config.trainer_config.items():
            setattr(self, key, val)
        self.client_config = train_config.client_config
        self.results_path = train_config.results_path
        self.trainer_type = train_config.trainer_type
        self.group = train_config.group
        
        self.swap_label = train_config.swap_label
        self.rounds = train_config.trainer_config['num_rounds']
        
        self.client_config = train_config.client_config
        self.results_path = train_config.results_path
        # Set the random set
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.writer = ResultWriter(train_config)
        
        self.central_model = []
        
        self.construct_model()
        # Construct the model
        
    
    def construct_model(self):
        clients, train_data, test_data = read_federated_data(self.dataset, group = self.group, swap_label = self.swap_label, central = True)
        model_path = 'flearn.model.%s.%s' % (self.dataset.split('_')[0], self.model)
        self.model_loader = importlib.import_module(model_path).construct_model
        client_model = self.model_loader(self.trainer_type, self.client_config['learning_rate'])
        self.clients = []
        for id in range(len(clients)):
            client_model = self.model_loader(self.trainer_type, self.client_config['learning_rate'])
            self.clients += [Client(id, self.client_config, train_data[clients[id]], test_data[clients[id]], \
                        model=client_model, validation = True)]
            
        
        
    
    def train(self):
        for round in range(self.rounds):
            print("Round ", round)
            test_results = dict()
            test_acc_client = []
            weights = []
            for c in self.clients:
                num_samples, train_acc, train_loss, soln, update = c.train()
                c.apply_update(update)
                #val_samples,val_acc, val_loss = c.validate()
                test_samples,test_acc, test_loss = c.test()
                print("Test Accuracy: ", test_acc, " with sampels ", test_samples)
                print("Train Accuracy: ", train_acc, " with sampels ", num_samples)
                test_results.update({c.id: test_acc})
                weights.append(test_samples)
                test_acc_client.append(test_acc)
                
            overall_acc = np.average(test_acc_client, weights = weights)
            self.writer.write(round, test_results, overall_acc)
        
        


# In[2]:


def test(dataset, model = 'mlp', swap_label = False, group = 1, seed = 2077):
    config = TrainConfig(dataset, model, 'Centralize', group = group, swap_label = swap_label, seed = seed)
    config.results_path = 'results/'+ dataset+ "/Centralize/"
    #config.trainer_config['num_rounds'] = 100
    if group == 1:
        config.results_path += "iid/"
    elif group != 1:
        config.results_path += "group_" + str(group) + "/"
    config.results_path += "seed" + str(seed) + "/"
    
    trainer = Centralize(config)
    trainer.train()



dataset = 'femnist'
model = 'mlp'
swap_label = 0 

group = 8
seed = 55
test(dataset = dataset,model = model, swap_label = swap_label, group = group, seed = seed)






