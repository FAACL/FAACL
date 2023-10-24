import numpy as np
from flearn.actor import Actor
import tensorflow as tf

class Server(Actor):
    def __init__(self, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, downlink=[], model=None,validation = False):
        actor_type = 'server'
        id = 0
        super(Server, self).__init__(id, actor_type, train_data, test_data, model,validation=validation)
        if len(downlink) > 0:
            self.add_downlink(downlink)
        self.refresh()

    def has_uplink(self):
        return False

    def check_trainable(self):

        if self.has_downlink():
            self.train_size = 0
            self.trainable = False
            for node in self.downlink:
                if node.check_trainable() == True:
                    self.trainable = True
                    self.train_size += node.train_size
        return self.trainable

    def check_testable(self):
        if self.has_downlink():
            self.test_size = 0
            self.testable = False
            for nodes in self.downlink:
                if nodes.check_testable() == True:
                    self.testable = True
                    self.test_size += nodes.test_size
        return self.testable

    def refresh(self):
        self.check_trainable()
        self.check_testable()

    def add_downlink(self, nodes):
        super(Server, self).add_downlink(nodes)
        self.refresh()

    def delete_downlink(self, nodes):
        super(Server, self).delete_downlink(nodes)
        self.refresh()

    def train(self, selected_nodes):
        results = []
        
        if self.downlink[0].actor_type == 'client':
            trainable, valid_nodes = self.check_selected_trainable(selected_nodes)
            if trainable == True:
                for node in valid_nodes:
                    num_samples, train_acc, train_loss, soln, update = node.train()
                    results.append([node, num_samples, train_acc, train_loss, update])
        
        elif self.downlink[0].actor_type == 'group':
            trainable, valid_nodes = self.check_selected_trainable(self.downlink)
            if trainable == True:
                for group in valid_nodes:
                    group_num_samples, group_train_acc, group_train_loss, soln, group_update = group.train(selected_nodes)
                    results.append([group, group_num_samples, group_train_acc, group_train_loss, group_update])
        
        if results == []:
            return
        return results
            

    def test(self, selected_nodes):
        testable, valid_nodes = self.check_selected_testable(selected_nodes)
        if testable == True:
            results = []
            for node in valid_nodes:
                num_samples, test_acc, test_loss = node.test()
                results.append([node, num_samples, test_acc, test_loss])
            return results
        else:
            return