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


# adative federated cluster learning
class FedDrift(object):
    def __init__(self, train_config, local_training, delta = 2):
        # Transfer trainer config to self, we save the configurations by this trick
        for key, val in train_config.trainer_config.items():
            setattr(self, key, val)
            
        self.local_training = local_training
        self.client_acc = False
        self.trainer_type = train_config.trainer_type
        self.group = train_config.group
        self.delta = delta
        self.results_path = train_config.results_path
        self.swap_label = train_config.swap_label
        # Get the config of client
        self.client_config = train_config.client_config
        # Get the config of group
        self.group_config = train_config.group_config
        if self.eval_locally == True:
            self.group_config.update({'eval_locally': True})
        else:
            self.group_config.update({'eval_locally': False})

        # Set the random set
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Construct the actors
        self.clients = []
        
        self.groups = []
        self.construct_actors()

        # Create results writer
        self.writer = ResultWriter(train_config)

        # Store the initial model params
        self.init_params = self.server.get_params()
        
        self.client_info = {}
        self.group_agginfo = {}
        self.agg_round = 5
        
    def initial_group_models(self):
        selected_client = random.sample(self.sample_clients, 1)[0]
        selected_client_model = selected_client.model
        empty_train_data, empty_test_data = {'x':[],'y':[]}, {'x':[],'y':[]}
        self.groups.append(Group(0, self.group_config, empty_train_data, empty_test_data,
                                [self.server], (selected_client_model)))
        
    
    def initial_client_models(self):
        for c in self.sample_clients:
            if c not in self.client_info:
                model_path = self.results_path + "client_"+ str(c.id) +"_local_" + str(self.local_training) + "_seed_" + str(self.seed) +  ".keras"
                if os.path.exists(model_path):
                    print("loading model " + str(c.id))
                    c.model = keras.models.load_model(model_path)
                    c.latest_params = c.model.get_weights()
                    val_samples, val_acc, val_loss = c.validate(list_form = True)
                else:
                    current_it = 0
                    while self.local_training > current_it:
                        current_it += 1
                        train_samples, train_acc, train_loss, soln, update = c.train()
                        c.apply_update(update)
                    val_samples, val_acc, val_loss = c.validate(list_form = True)
                    c.model.save(model_path) 
                    print("saving model " + str(c.id))
                self.client_info.update({c: [val_samples, val_acc, val_loss]})

    def construct_actors(self):
        # 1, Read dataset
        clients, train_data, test_data = read_federated_data(self.dataset, group = self.group, swap_label = self.swap_label)

        # 2, Get model loader according to dataset and model name and construct the model
        # Set the model loader according to the dataset and model name
        model_path = 'flearn.model.%s.%s' % (self.dataset.split('_')[0], self.model)
        self.model_loader = importlib.import_module(model_path).construct_model
        # Construct the model
        client_model = self.model_loader(self.trainer_type, self.client_config['learning_rate'])

        # 3, Construct server
        self.server = Server(model=client_model)
        
        self.clients = []
        for id in range(len(clients)):
            client_model = self.model_loader(self.trainer_type, self.client_config['learning_rate'])
            self.clients += [Client(id, self.client_config, train_data[clients[id]], test_data[clients[id]], \
                        model=client_model, validation = True)]

        # 6, Set the server's downlink to groups
        self.server.add_downlink(self.groups)

        # 7*, We evaluate the auxiliary global model on server
        # To speed the testing, we need construct a local test dataset for server
        if self.eval_global_model == True:
            server_test_data = {'x':[], 'y':[]}
            for c in clients:
                server_test_data['x'].append(test_data[c]['x'])
                server_test_data['y'].append(test_data[c]['y'])
            self.server.test_data['x'] = np.vstack(server_test_data['x'])
            self.server.test_data['y'] = np.hstack(server_test_data['y'])

    

    def train(self):    
        self.sample_clients = self.clients
        self.initial_client_models()
        idx=0
        cluster = dict()
        for client in self.sample_clients:
            empty_train_data, empty_test_data = {'x':[],'y':[]}, {'x':[],'y':[]}
            new_group = Group(idx, self.group_config, empty_train_data, empty_test_data,
                                [self.server], (client.model))
            self.groups.append(new_group)
            idx += 1
            cluster.update({new_group: [client]})
        
        # Update cluster identity 
        for r in range(300):
            
            start_time0 = time.time()
            print("Iteration ", r, " Number of groups ", len(self.groups))
            if r > 0:
                cluster = dict()
                for client in self.sample_clients:
                    low_loss = None
                    pick_group = None
                    for group in self.groups:
                        client.latest_params = group.model.get_weights()
                        num_samples, train_acc, train_loss, soln, update = client.train()
                        if low_loss is None:
                            low_loss = train_loss
                            pick_group = group
                        elif low_loss > train_loss:
                            low_loss = train_loss
                            pick_group = group
                    if pick_group in cluster:
                        client_lst = cluster[pick_group] + [client]
                        cluster.update({pick_group: client_lst})
                    else:
                        cluster.update({pick_group: [client]})
                        

            # Test group 1 model on group 2 client
            Loss = dict()
            for group1 in self.groups:
                Loss_g1 = dict()
                for group2 in self.groups:
                    Loss_client = []
                    if group2 not in cluster:
                        continue
                    for client in cluster[group2]:
                        client.latest_params = group1.model.get_weights()
                        num_samples, train_acc, train_loss, soln, update = client.train()
                        Loss_client.append(train_loss)
                    Loss_g1.update({group2: sum(Loss_client)})
                Loss.update({group1: Loss_g1})

            groups_to_merge = dict()
            for group1 in cluster.keys():
                if group1 in groups_to_merge.keys() or group1 in groups_to_merge.values():
                    continue
                for group2 in cluster.keys():
                    if group1 == group2:
                        continue
                    if group2 in groups_to_merge.keys() or group2 in groups_to_merge.values():
                        continue

                    dist = max(Loss[group1][group2] - Loss[group1][group1],\
                               Loss[group2][group1] - Loss[group2][group2],0)
                    if dist < self.delta:
                        groups_to_merge.update({group1: group2})

            group_to_add = []
            for group1, group2 in groups_to_merge.items():
                weight1 = 0
                weight2 = 0
                for c in cluster[group1]:
                    weight1 += c.train_size
                for c in cluster[group2]:
                    weight2 += c.train_size
                weights = [weight1, weight2]
                weights = np.array(weights, dtype=float) / np.sum(weights, dtype=np.float)
                updates = [group1.model.get_weights(), group2.model.get_weights()]
                num_clients = len(updates)
                num_layers = len(updates[0])
                agg_updates = []
                for l in range(num_layers):
                    agg_updates.append(np.sum([up[l]*w for up, w in zip(updates, weights)], axis=0))
                new_model = group1.model
                new_model.set_weights(agg_updates)
                new_group = Group(idx, self.group_config, empty_train_data, empty_test_data,
                                    [self.server], new_model)
                idx += 1
                self.groups.append(new_group)
                cluster[new_group] = cluster[group1] + cluster[group2]
                self.groups.remove(group1)
                self.groups.remove(group2)
                cluster.pop(group1)
                cluster.pop(group2)
            
            train_time = round(time.time() - start_time0, 3)
            print("Merge Time ", train_time)
            start_time_train = time.time()
            print(len(self.groups))
            num_round = 1
            for it in range(num_round):
                client_acc = dict()
                test_results = dict()
                for group in cluster.keys():
                    updates = []
                    weights = []
                    test_Acc = []
                    test_Weight = []
                    for client in cluster[group]:
                        client.latest_params = group.model.get_weights()
                        test_samples,test_acc, test_loss = client.test()
                        val_samples,val_acc, val_loss = client.test()
                        num_samples, train_acc, train_loss, soln, update = client.train()
                        updates.append([(w0+up) for up, w0 in zip(update, client.latest_params)])
                        weights.append(num_samples)
                        test_Acc.append(test_acc)
                        test_Weight.append(test_samples)
                        
                        if client not in client_acc.keys():
                            client_acc.update({client: [test_acc, test_samples, val_acc]})
                        elif client_acc[client][2] < val_acc:
                            client_acc.update({client: [test_acc, test_samples, val_acc]})
                    weights = np.array(weights, dtype=float) / np.sum(weights, dtype=np.float)
                    num_clients = len(updates)
                    num_layers = len(updates[0])
                    agg_updates = []
                    for l in range(num_layers):
                        agg_updates.append(np.sum([up[l]*w for up, w in zip(updates, weights)], axis=0))
                    group.model.set_weights(agg_updates)
                    test_acc_round_t = np.average(test_Acc, weights = test_Weight)
                    print("Round " + str(it) + " cluster " +  str(group.id) + " test acc: " + str(test_acc_round_t))
                    lst = []
                    for c in cluster[group]:
                        lst.append(c.id)
                    test_results.update({tuple(lst): test_acc_round_t})
                    print("with clients: ", lst)
                weight = []
                test_acc = []
                for t_acc, t_samples, v_acc in client_acc.values():
                    weight.append(t_samples)
                    test_acc.append(t_acc)
                overall_acc = np.average(test_acc, weights = weight)
                print("Overall acc ", overall_acc)
                self.writer.write(r, test_results, overall_acc)
                
            for g in self.groups:
                if g not in cluster:
                    self.groups.remove(g)
            train_time = round(time.time() - start_time_train, 3)
            print("Model Trainging Time ", train_time)
            train_time = round(time.time() - start_time0, 3)
            print("Total Training Time ", train_time)
            self.writer.write(r, time = [train_time])
            

    def select_clients(self, comm_round=1, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
            For the consideration of test comparability, we first select the client by round robin, and then select by randomly
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''

        #num_clients = min(num_clients, len(self.clients))
        # Round robin
        """
        if comm_round < len(self.clients) / num_clients:
            head = comm_round * num_clients
            if head + num_clients <= len(self.clients):
                selected_clients = self.clients[head: head+num_clients]
            else:
                selected_clients = self.clients[head:] + self.clients[:head+num_clients-len(self.clients)]
        # Random selecte clients
        else:
            random.seed(comm_round+self.seed)  # make sure for each comparison, we are selecting the same clients each round
            selected_clients = random.sample(self.clients, num_clients)
            random.seed(self.seed) # Restore the seed
        random.seed(comm_round+self.seed)  # make sure for each comparison, we are selecting the same clients each round
        selected_clients = random.sample(self.clients, num_clients)
        random.seed(self.seed) # Restore the seed
        return selected_clients
        
        """
        random.seed(comm_round+self.seed)  # make sure for each comparison, we are selecting the same clients each round
        selected_clients = random.sample(self.clients, num_clients)
        random.seed(self.seed) # Restore the seed
        return selected_clients
    
    def federated_averaging_aggregate(self, updates, nks):
        return self.weighted_aggregate(updates, nks)

    def simply_averaging_aggregate(self, params_list):
        weights = [1.0] * len(params_list)
        return self.weighted_aggregate(params_list, weights)

    def weighted_aggregate(self, updates, weights):
        # Aggregate the updates according their weights
        normalws = np.array(weights, dtype=float) / np.sum(weights, dtype=np.float)
        num_clients = len(updates)
        num_layers = len(updates[0])
        agg_updates = []
        for la in range(num_layers):
            agg_updates.append(np.sum([up[la]*pro for up, pro in zip(updates, normalws)], axis=0))

        return agg_updates # -> list

    '''
    Summary the train results or test results
    '''
    def summary_results(self, comm_round, train_results=None, test_results=None):

        partial_test_acc = False
        ty2 = ''
        if train_results:
            results = train_results
            ty, cor = 'Train', 'blue'
        elif test_results:
            results = test_results
            ty, cor = 'Test', 'red'
            if results[0][0].actor_type == 'server':
                ty, cor = 'Auxiliary Model Test', 'green'
            if comm_round < len(self.clients) / min(self.clients_per_round, len(self.clients)):
                ty2 += '(Partial)'
                # We do not write the partial test accuracy
                partial_test_acc = True
            else:
                ty2 += '(Complete)'
        else:
            return

        nks = [rest[1] for rest in results]
        num_sublink = len(nks) # Groups or clients
        accs = [rest[2] for rest in results]
        losses = [rest[3] for rest in results]
        weighted_acc = np.average(accs, weights=nks)
        weighted_loss = np.average(losses, weights=nks)
        print(colored(f'Round {comm_round}, {ty+ty2} ACC: {round(weighted_acc, 4)},\
            {ty+ty2} Loss: {round(weighted_loss, 4)}', cor, attrs=['reverse']))

        summary = {'Total': (num_sublink, weighted_acc, weighted_loss)}
        # Clear partial test result on summary
        #if partial_test_acc == True: summary = {'Total': (None, None, None)}

        # Record group accuracy and loss
        """
        if results[0][0].actor_type == 'group':
            groups = [rest[0] for rest in results]
            for idx, g in enumerate(groups):
                if partial_test_acc == True: accs[idx] 
                summary[f'G{g.id}'] = (accs[idx], losses[idx], nks[idx]) # accuracy, loss, number of samples
                print(f'Round {comm_round}, Group: {g.id}, {ty} ACC: {round(accs[idx], 4)},\
                    {ty} Loss: {round(losses[idx], 4)}')

                # Clear partial group test result on summary
                if partial_test_acc == True: summary[f'G{g.id}'] = (None, None, None)
        """

        return summary

    def train_locally(self, num_epoch=20, batch_size=10):
        """
            We can train and test model on server for comparsion or debugging reseason
        """
        # 1, We collect all data into server
        print("Collect data.....")
        server_test_data = {'x':[], 'y':[]}
        server_train_data = {'x':[], 'y':[]}
        for c in self.clients:
            server_test_data['x'].append(c.test_data['x'])
            server_test_data['y'].append(c.test_data['y'])
            server_train_data['x'].append(c.train_data['x'])
            server_train_data['y'].append(c.train_data['y'])
        self.server.test_data['x'] = np.vstack(server_test_data['x'])
        self.server.test_data['y'] = np.hstack(server_test_data['y'])
        self.server.train_data['x'] = np.vstack(server_train_data['x'])
        self.server.train_data['y'] = np.hstack(server_train_data['y'])

        self.server.model.summary()

        # 2, Server train locally
        train_size, train_acc, train_loss, soln, update = self.server.solve_inner(num_epoch, batch_size)
        # 3, Server Apply update
        self.server.apply_update(update)
        # 4, Server test locally
        test_size, test_acc, test_loss = self.server.test_locally()

        # 5, Print result, we show the accuracy and loss of all training epochs
        print(f"Train size: {train_size} Train ACC: {[round(acc, 4) for acc in train_acc]} \
             Train Loss: {[round(loss, 4) for loss in train_loss]}")
        print(colored(f"Test size: {test_size}, Test ACC: {round(test_acc, 4)}, \
            Test Loss: {round(test_loss, 4)}", 'red', attrs=['reverse']))

        return

    def weighted_aggregate(self, updates, weights):
        # Aggregate the updates according their weights
        epsilon = 1e-5 # Prevent divided by 0
        normalws = np.array(weights, dtype=float) / (np.sum(weights, dtype=np.float) + epsilon)
        num_layers = len(updates[0])
        agg_updates = []
        for la in range(num_layers):
            agg_updates.append(np.sum([up[la]*pro for up, pro in zip(updates, normalws)], axis=0))

        return agg_updates # -> list


    # Refresh the difference value (discrepancy) and cosine dissimilarity cosine of clients and gorups and
    # return the discrepancy (w/0 cosine dissimilarity) information for summary
    def refresh_discrepancy_and_dissmilarity(self, clients):
        def _calculate_mean_diffs(clients):
            discrepancy = [c.discrepancy for c in clients]
            dissimilarity = [c.cosine_dissimilarity for c in clients]
            return np.mean(discrepancy), np.mean(dissimilarity)

        # Call the discrepancy update function of clients
        for c in clients: c.update_difference()

        diffs = {}
        diffs['Total'] = _calculate_mean_diffs(clients)[0] # Return discrepancy
        groups = set([c.uplink[0] for c in clients])
        for g in groups:
            gc = [c for c in clients if c.uplink[0] == g]
            g.discrepancy, g.cosine_dissimilarity = _calculate_mean_diffs(gc)
            # i.e. { 'G1': (numer of group clients, discrepancy) }
            diffs[f'G{g.id}'] = (len(gc), g.discrepancy)
        return diffs

    
#!/usr/bin/env python
# coding: utf-8

# In[1]:

def test(dataset, local_training, model = 'mlp', swap_label = False, group = 1, seed = 2077,epsilon_wx = 0.3, delta = 2):
    config = TrainConfig(dataset, model, 'FedDrift', group = group, swap_label = swap_label, seed = seed)
    config.results_path = 'results/'+ dataset+ "/FedDrift/"

    if group == 1:
        config.results_path += "iid/"
    elif group != 1:
        config.results_path += "group_" + str(group) + "/"
    
    config.results_path += "local_training" + str(local_training) + "/"
    config.results_path += "seed" + str(seed) + "/"
    trainer = FedDrift(config,local_training, delta = delta)
    trainer.train()
    
delta = 3

Dataset = ['mnist', 'emnist', 'fmnist']
Model = ['mlp']
Group = [1,2,3,6,7,8] 
Seed = [10, 55, 2077]

for dataset in Dataset:
    for model in Model:
        for group in Group:
            for seed in Seed:
                test(dataset=dataset,model = model, local_training=50, group=group, seed = seed, delta = delta)


delta = 4
dataset = 'femnist'
model = 'mlp'
group = 0
Seed = [10, 55, 2077]
for seed in Seed:
    test(dataset=dataset,model = model, local_training=50, group=group, seed = seed, delta = delta)





