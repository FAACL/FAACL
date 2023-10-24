from flearn.actor import Actor
import numpy as np
from math import floor

class Group(Actor):
    def __init__(self, id, config, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, uplink=[], model=None,validation = False):
        actor_type = 'group'
        super(Group, self).__init__(id, actor_type, train_data, test_data, model=model, validation=validation)
        if len(uplink) > 0:
            self.add_uplink(uplink)
        for key, val in config.items(): 
            setattr(self, key, val)

        self.discrepancy = 0 
        self.cosine_dissimilarity = 0

        self.opt_updates = None
        self.aggregation_strategy = 'fedavg'

    def check_trainable(self):
        self.trainable = False
        if self.has_downlink():
            self.train_size = 0
            for node in self.downlink:
                if node.check_trainable() == True:
                    self.trainable = True
                    self.train_size += node.train_size
        return self.trainable

    def check_testable(self):
        self.testable = False
        if self.has_downlink():
            self.test_size = 0
            for nodes in self.downlink:
                if nodes.check_testable() == True:
                    self.testable = True
                    self.test_size += nodes.test_size
        return self.testable

    def refresh(self):
        self.check_trainable()
        self.check_testable()
        if self.eval_locally == True:
            if self.downlink:
                group_test_data = {'x':[], 'y':[]}
                for c in self.downlink:
                    group_test_data['x'].append(c.test_data['x'])
                    group_test_data['y'].append(c.test_data['y'])
                self.test_data['x'] = np.vstack(group_test_data['x'])
                self.test_data['y'] = np.hstack(group_test_data['y'])
            else:
                self.test_data = {'x':[], 'y':[]}
        return

    def add_downlink(self, nodes):
        super(Group, self).add_downlink(nodes)
        self.refresh()

    def delete_downlink(self, nodes):
        super(Group, self).delete_downlink(nodes)
        self.refresh()

    def clear_downlink(self):
        super(Group, self).clear_downlink()
        self.refresh()
    
    def federated_averaging_aggregate(self, updates, nks):
        return self.weighted_aggregate(updates, nks)

    def federated_averaging_aggregate_with_temperature(self, updates, nks, temps, max_temp):
        if len(temps) == 0: 
            return [np.zeros_like(ws) for ws in self.latest_params]
        else:
            temp_nks, epsilon = [], 1e-5
            for nk, temp in zip(nks, temps):
                if temp == None:
                    temp_nks.append(nk)
                else:
                    temp_nks.append(floor((max(temp, 0) / (max_temp+epsilon)) * nk))
            return self.federated_averaging_aggregate(updates, temp_nks)

    def weighted_aggregate(self, updates, weights):
        epsilon = 1e-5 
        normalws = np.array(weights, dtype=float) / (np.sum(weights, dtype=np.float) + epsilon)
        num_layers = len(updates[0])
        agg_updates = []
        for la in range(num_layers):
            agg_updates.append(np.sum([up[la]*pro for up, pro in zip(updates, normalws)], axis=0))

        return agg_updates

    def _calculate_weighted_metric(metrics, nks):
            normalws = np.array(nks) / np.sum(nks, dtype=np.float)
            metric = np.sum(metrics*normalws)
            return metric

    def train(self, selected_nodes=None):
        
        if len(self.downlink) == 0:
            print(f"Warning: Group {self.id} is empty.")
            return 0, 0, 0, None

        if not selected_nodes: selected_nodes = self.downlink

        trainable, valid_nodes = self.check_selected_trainable(selected_nodes)
        
        if trainable == True:
            train_results = []
            group_params = self.latest_params

            for node in valid_nodes:
                node.latest_updates = [(w1-w0) for w0, w1 in zip(node.latest_params, group_params)]
                node.latest_params = group_params
            
            for node in valid_nodes:
                num_samples, train_acc, train_loss, soln, update = node.train()
                train_results.append([node, num_samples, train_acc, train_loss, update])
            
            nks = [rest[1] for rest in train_results] 
            updates = [rest[4] for rest in train_results] 
            temps = [rest[0].temperature for rest in train_results]
            max_temp = train_results[0][0].max_temp
            if self.aggregation_strategy == 'temp' and max_temp is not None:
                agg_updates = self.federated_averaging_aggregate_with_temperature(updates, nks, temps, max_temp)
            if self.aggregation_strategy == 'fedavg':
                agg_updates = self.federated_averaging_aggregate(updates, nks)
            if self.aggregation_strategy == 'avg':
                agg_updates = self.federated_averaging_aggregate(updates, [1.0*len(nks)])

            self.fresh_latest_params_updates(agg_updates)
           

           
            group_num_samples = np.sum(nks, dtype=np.float)
            group_train_acc = np.average([rest[2] for rest in train_results], weights=nks)
            group_train_loss = np.average([rest[3] for rest in train_results], weights=nks)

            return group_num_samples, group_train_acc, group_train_loss, self.latest_params, agg_updates

        elif self.allow_empty == True:
            group_num_samples, group_train_acc, group_train_loss, update = 0, 0, 0, None
            return group_num_samples, group_train_acc, group_train_loss, self.latest_params, update
        else:
            print(f'ERROR: Group {self.id} has not any valid training clients with training data which is invalid.')
            return
    def test(self):
        if len(self.downlink) == 0:
            print(f"Warning: Group {self.id} is empty.")
            return 0, 0, 0
        
        testable, valid_nodes = self.check_selected_testable(self.downlink)
        if testable == False:
            print(f'Warning: Group {self.id} has not test data.')
            return 0, 0, 0

        if self.eval_locally == False:
            test_results = [node.test() for node in valid_nodes]
            nks = [rest[0] for rest in test_results]
            group_num_samples = np.sum(nks, dtype=np.float)
            group_test_acc = np.average([rest[1] for rest in test_results], weights=nks)
            group_test_loss = np.average([rest[2] for rest in test_results], weights=nks)
        else:
            group_num_samples, group_test_acc, group_test_loss = self.test_locally()
        
        return group_num_samples, group_test_acc, group_test_loss