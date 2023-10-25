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
import time
def fun(ele):
    return ele.id
def sig(x):
    return 1/(1 + np.exp(-x))
# adative federated cluster learning
class FAACL(object):
    def __init__(self, train_config, local_training,epsilon_wx):
        self.epsilon_wx = epsilon_wx
        for key, val in train_config.trainer_config.items():
            setattr(self, key, val)
        self.local_training = local_training
        self.trainer_type = train_config.trainer_type
        self.group = train_config.group
        # Get the config of client
        self.client_config = train_config.client_config
        self.results_path = train_config.results_path
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
        self.writer = ResultWriter(train_config, epsilon = self.epsilon_wx)

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
        clients, train_data, test_data = read_federated_data(self.dataset, group = self.group)

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
        clustering = dict()
        partition = []
        num_round = 10
        t=2
        num_round = 1
        # Initial partition
        for c in self.sample_clients:
            p = tuple([c])
            partition.append(p)
            clustering.update({p: {tuple([c]): c.model}})
        while len(partition) > 1:
            train_time, test_time, agg_time = 0, 0, 0
            new_partition = []
            new_clustering = dict()
            for i in range(0, len(partition), t):
                if i+t <= len(partition):
                    #print("Case 1")
                    new_p = []
                    new_d = dict()
                    k = 0
                    while k < t:
                        new_p += list(partition[i + k])
                        new_d.update(clustering[partition[i + k]])
                        k += 1
                    #print("New partition: ", list(map(fun, new_p)))
                    new_p.sort(key=fun)
                    new_p = tuple(new_p)
                    new_partition.append(new_p)
                    new_clustering.update({new_p: new_d})
                else:
                    #print("Case 2")
                    new_p = []
                    new_d = dict()
                    k = 0
                    while k < len(partition) - i:
                        new_p += list(partition[i + k])
                        new_d.update(clustering[partition[i + k]])
                        k += 1
                    #print("New partition: ", list(map(fun, new_p)))
                    new_p.sort(key=fun)
                    new_p = tuple(new_p)
                    new_partition.append(new_p)
                    new_clustering.update({new_p: new_d})
            
            # Cluster merge
            start_time0 = time.time()
            
            for _ in range(5):
                start_time = time.time()
                for p in new_partition:
                    print("Current partition is ", list(map(fun, list(p))))
                    c = new_clustering[p]
                    print("With clustering ")
                    for clu in c.keys():
                        print(list(map(fun, list(clu))))
                    # first test each cluster model on each client and record their val loss & val acc
                    client_loss = dict()
                    client_acc = dict()
                    client_weight = dict()
                    
                    for client in p:
                        loss_c = dict()
                        acc_c = dict()
                        weight_c = dict()
                        for cluster, model in c.items():
                            client.latest_params = model.get_weights()
                            val_samples, val_acc, val_loss = client.validate(list_form = True)
                            loss_c.update({cluster: val_loss})
                            acc_c.update({cluster: val_acc})
                            weight_c.update({cluster: val_samples})
                        client_loss.update({client: loss_c})
                        client_acc.update({client: acc_c})
                        client_weight.update({client: weight_c})
                    
                    
                    cluster_to_merge = dict()
                    cluster_loss = dict()
                    for k, v in c.items():
                        client_loss_tmp = []
                        client_weight_tmp = []
                        for client in k:
                            loss = client_loss[client][k]
                            weight = client_weight[client][k]
                            client_loss_tmp.append(sum(loss))
                            client_weight_tmp.append(weight)
                        cluster_loss.update({k: np.average(np.array(client_loss_tmp), weights = np.array(client_weight_tmp))})
                    
                    for k1, v1 in c.items():
                        for k2, v2 in c.items():
                            if set(k1).union(set(k2)) == set(k1):
                                continue
                            # for each client in k1, test v1 and v2
                            flag_merge = True
                            for client in k1:
                                l1 = client_loss[client][k1]
                                acc1 = client_acc[client][k1]
                                # loss from other cluster model
                                l2 = client_loss[client][k2]
                                acc2 = client_acc[client][k2]

                                # construct corrected loss
                                cor_loss1 = []
                                cor_loss2 = []
                                cor_loss1_epsilon = []
                                for k in range(len(acc1)):
                                    if acc1[k] == 1:
                                        cor_loss1.append(l1[k])
                                    else:
                                        if self.dataset == 'femnist62':
                                            cor_loss1.append(4.13)
                                        else:
                                            cor_loss1.append(2.3)

                                for k in range(len(acc2)):
                                    if acc2[k] == 1:
                                        cor_loss2.append(l2[k])
                                    else:
                                        if self.dataset == 'femnist62':
                                            cor_loss2.append(4.13)
                                        else:
                                            cor_loss2.append(2.3)

                                for k in range(len(acc1)):
                                    if acc1[k] == 1:
                                        cor_loss1_epsilon.append(l1[k] + self.epsilon_wx)
                                    else:
                                        if self.dataset == 'femnist62':
                                            cor_loss1_epsilon.append(4.13)
                                        else:
                                            cor_loss1_epsilon.append(2.3)
                                # Apply statistical test
                                # paired t-test: pval = stats.ttest_rel(cor_loss1_epsilon, cor_loss2, alternative = 'greater').pvalue
                                # wilcoxon signed rank test
                                pval = stats.wilcoxon(cor_loss1_epsilon, cor_loss2, alternative = 'greater').pvalue
                                if pval > 0.1:
                                    print("Cluster fail to merge with pvalue ", pval)
                                    print(list(map(fun, k1)))
                                    print(list(map(fun, k2)))
                                    flag_merge = False
                                    break
                            if flag_merge:
                                print("Start merging")
                                print(list(map(fun, k1)))
                                print(list(map(fun, k2)))
                                if k1 in cluster_to_merge:
                                    new_lst = cluster_to_merge[k1] + [k2]
                                    cluster_to_merge.update({k1: new_lst})
                                else:
                                    cluster_to_merge.update({k1: [k2]})
                    
                    cluster_to_remove = []
                    cluster_to_add = dict()
                    for cluster1, cluster_list in cluster_to_merge.items():
                        newcluster = list(cluster1)
                        for cluster in cluster_list:
                            newcluster = list(set(newcluster + list(cluster)))
                        newcluster.sort(key=fun)
                        newcluster = tuple(newcluster)
                        if newcluster not in c.keys():
                            cluster_to_add.update({newcluster: c[cluster1]})
                            
                        cluster_to_remove.append(cluster1)
                    for cluster in cluster_to_remove:
                        c.pop(cluster)
                    c.update(cluster_to_add)
                    new_clustering[p] = c
                
                merge_time = round(time.time() - start_time, 3)
                print("Test all cluster models on all clients  Time ", merge_time)
                start_time = time.time()
                # Then start training each cluster
                for it in range(num_round):
                    client_acc = dict()
                    test_results = dict()
                    for p in new_partition:
                        for cluster, model in new_clustering[p].items():
                            updates = []
                            weights = []
                            test_Acc = []
                            test_Weight = []
                            for client in cluster:
                                client.latest_params = model.get_weights()
                                
                                val_sample, val_acc, val_loss = client.validate()
                                test_samples,test_acc, test_loss = client.test()
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
                            model.set_weights(agg_updates)
                            test_acc_round_t = np.average(test_Acc, weights = test_Weight)
                            test_results.update({tuple(list(map(fun, list(cluster)))): test_acc_round_t})
                            print("Round " + str(it) + " cluster " +  str(list(map(fun, list(cluster)))) + " test acc: " + str(test_acc_round_t))
                    weight = []
                    test_acc = []
                    for (t_acc, t_samples, v_acc) in client_acc.values():
                        weight.append(t_samples)
                        test_acc.append(t_acc)
                    overall_acc = np.average(test_acc, weights = weight)
                    print("Overall Acc: ", overall_acc)
                    self.writer.write(round=it, test_results=test_results, overall_acc=overall_acc)
            
                    
            train_time = round(time.time() - start_time, 3)
            print("Cluster models training ", train_time)
            total_time = round(time.time() - start_time0, 3)
            print("Total Training Time ", total_time)
            self.writer.write(time=[train_time, merge_time, total_time])
            partition = new_partition
            clustering = new_clustering
            for c in clustering:
                print(list(map(fun, list(c))))
                
        # Keep training until convergence
        continue_round = 300
        if self.model == 'cnn':
            continue_round = 100
            
        for it in range(continue_round):
            start_time = time.time()
            client_acc = dict()
            test_results = dict()
            for p in partition:
                for cluster, model in clustering[p].items():
                    updates = []
                    weights = []
                    test_Acc = []
                    test_Weight = []
                    for client in cluster:
                        client.latest_params = model.get_weights()
                        val_sample, val_acc, val_loss = client.validate()
                        test_samples,test_acc, test_loss = client.test()
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
                    model.set_weights(agg_updates)
                    test_acc_round_t = np.average(test_Acc, weights = test_Weight)
                    test_results.update({tuple(list(map(fun, list(cluster)))): test_acc_round_t})
                    print("Round " + str(it) + " cluster " +  str(list(map(fun, list(cluster)))) + " test acc: " + str(test_acc_round_t))
            weight = []
            test_acc = []
            for (t_acc, t_samples, v_acc) in client_acc.values():
                weight.append(t_samples)
                test_acc.append(t_acc)
            overall_acc = np.average(test_acc, weights = weight)
            print("Overall Acc: ", overall_acc)
            
            train_time = round(time.time() - start_time, 3)
            print("Training Time at iteration ",it, " is ", train_time)
            self.writer.write(round=it, test_results=test_results, overall_acc=overall_acc, time = [train_time])
    def select_clients(self, comm_round=1, num_clients=20):
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

def test(dataset, local_training, \
         model = 'mlp', group = 1, seed = 2077,epsilon_wx = 0.3):
    config = TrainConfig(dataset, model, 'AdaFCL', group = group, seed = seed)
    config.results_path = 'results/'+ dataset+ "/FAACL/"

    if group == 1:
        config.results_path += "iid/"
    elif group != 1:
        config.results_path += "group_" + str(group) + "/"
    
    config.results_path += "local_training" + str(local_training) + "/"
    config.results_path += "seed" + str(seed) + "/"
    
    
    trainer = FAACL(config,local_training, epsilon_wx = epsilon_wx)
    trainer.train()


# In[3]:
# Finish:

# Running FAACL on mnist iid with epsilon 0.3 with seed 2077

dataset = 'mnist'
epsilon_wx = 0.3
model = 'mlp'
local_training = 50
seed = 2077
group = 1
test(dataset=dataset, local_training = local_training, model = model, epsilon_wx = epsilon_wx, group = group, seed = seed)







