import numpy as np

import random
from math import ceil


class Actor(object):
    def __init__(self, id, actor_type, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, model=None, validation=False):
        self.validation = validation
        size = len(train_data['x'])
        random.seed(0)
        np.random.seed(0)
        p_train = np.random.permutation((size))
        train_data['x'] = np.array(train_data['x'])[p_train]
        train_data['y'] = np.array(train_data['y'])[p_train]
        p_test = np.random.permutation(len(test_data['x']))
        test_data['x'] = np.array(test_data['x'])[p_test]
        test_data['y'] = np.array(test_data['y'])[p_test]
        if self.validation:
            idx = random.sample(range(0,size), int (size * 0.25)+1)
            self.val_data = {'x':[],'y':[]}
            self.val_data['x'] = np.array([train_data['x'][i] for i in idx])
            self.val_data['y'] = np.array([train_data['y'][i] for i in idx])
            rest = list(set(list(range(0,size))).symmetric_difference(set(idx)))
            self.train_data = {'x':[],'y':[]}
            self.train_data['x'] = np.array([train_data['x'][i] for i in rest])
            self.train_data['y'] = np.array([train_data['y'][i] for i in rest])
        else:  
            self.train_data = train_data
        self.id = id
        self.test_data = test_data
        self.model = model
        self.actor_type = actor_type
        self.name = 'NULL'
        self.latest_params, self.latest_updates = None, None
        self.local_soln, self.local_gradient = None, None
        self.train_size, self.test_size = 0, 0 
        self.uplink, self.downlink = [], []
        self.trainable, self.testable, self.valable = False, False, False

        self.preprocess()

    def preprocess(self):
        self.name = str(self.actor_type) + str(self.id)
        self.latest_params, self.local_soln = self.get_params(), self.get_params()
        self.latest_updates = [np.zeros_like(ws) for ws in self.latest_params]
        self.local_gradient = [np.zeros_like(ws) for ws in self.latest_params]

    '''Return the parameters of global model instance
    '''
    def get_params(self):
        if self.model:
            return self.model.get_weights()
    
    def set_params(self, weights):
        if self.model:
            self.model.set_weights(weights)

    def solve_inner(self, num_epoch=1, batch_size=10, pretrain=False, amount = None):
        if self.train_data['y'].shape[0] > 0:
            X, y_true = self.train_data['x'], self.train_data['y']
            if amount and amount <= len(train_data['x']):
                X, y_true = self.train_data['x'][:amount], self.train_data['y'][:amount]
            num_samples = y_true.shape[0]
            backup_params = self.get_params()
            t0_weights = self.latest_params
            self.set_params(t0_weights)
            history = self.model.fit(X, y_true, batch_size, num_epoch, verbose=0)
            t1_weights = self.get_params()
            gradient = [(w1-w0) for w0, w1 in zip(t0_weights, t1_weights)]
            
            self.set_params(backup_params)
            if pretrain == False:
                self.local_soln = t1_weights
                self.local_gradient = gradient
            train_acc = history.history['accuracy']
            train_loss = history.history['loss']
            return num_samples, train_acc, train_loss, t1_weights, gradient
        else:
            return 0, [0], [0], self.latest_params, [np.zeros_like(ws) for ws in self.latest_params]

    def solve_iters(self, num_iters=1, batch_size=10, pretrain=False):

        def batch_data_multiple_iters(data, batch_size, num_iters):
            data_x = data['x']
            data_y = data['y']
            data_size = data_y.shape[0]

            random_idx = np.arange(data_size)
            np.random.shuffle(random_idx)
            data_x, data_y = data_x[random_idx], data_y[random_idx]
            max_iter = ceil(data_size / batch_size)

            for iter in range(num_iters):
                round_step = (iter+1) % max_iter
                if round_step == 0:
                    x_part1, y_part1 = data_x[(max_iter-1)*batch_size: data_size], \
                        data_y[(max_iter-1)*batch_size: data_size]
                    np.random.shuffle(random_idx)
                    data_x, data_y = data_x[random_idx], data_y[random_idx]
                    x_part2, y_part2 = data_x[0: max_iter*batch_size%data_size], \
                        data_y[0: max_iter*batch_size%data_size]

                    batched_x = np.vstack([x_part1, x_part2])
                    batched_y = np.hstack([y_part1, y_part2])  
                else:
                    batched_x = data_x[(round_step-1)*batch_size: round_step*batch_size]
                    batched_y = data_y[(round_step-1)*batch_size: round_step*batch_size]

                yield (batched_x, batched_y)

        num_samples = self.train_data['y'].shape[0]
        if num_samples == 0:
            return 0, [0], [0], self.latest_params, [np.zeros_like(ws) for ws in self.latest_params]

        backup_params = self.get_params()
        t0_weights = self.latest_params
        self.set_params(t0_weights)
        train_results = []
        for X, y in batch_data_multiple_iters(self.train_data, batch_size, num_iters):
            train_results.append(self.model.train_on_batch(X, y))
        t1_weights = self.get_params()
        gradient = [(w1-w0) for w0, w1 in zip(t0_weights, t1_weights)]
        self.set_params(backup_params)
        if pretrain == False:
            self.local_soln = t1_weights
            self.local_gradient = gradient
        train_acc = [rest[1] for rest in train_results]
        train_loss = [rest[0] for rest in train_results]
        
        return num_samples, train_acc, train_loss, t1_weights, gradient

    def apply_update(self, update):
        t0_weights = self.get_params()
        t1_weights = [(w0+up) for up, w0 in zip(update, t0_weights)]
        self.set_params(t1_weights)
        self.latest_updates = update
        self.latest_params = t1_weights
        return self.latest_params
    
    def fresh_latest_params_updates(self, update):
        prev_params = self.latest_params
        latest_params = [(w0+up) for up, w0 in zip(update, prev_params)]
        self.latest_updates = update
        self.latest_params = latest_params
        return self.latest_params, self.latest_updates
    
    def test_locally(self):
        if self.test_data['y'].shape[0] > 0:
            backup_params = self.get_params()
            self.set_params(self.latest_params)
            X, y_true = self.test_data['x'], self.test_data['y']
            loss, acc = self.model.evaluate(X, y_true, verbose=0)
            self.set_params(backup_params)
            return self.test_data['y'].shape[0], acc, loss
        else:
            return 0, 0, 0
     
    def val_locally(self, list_form = False):
        if self.val_data['y'].shape[0] > 0:
            backup_params = self.get_params()
            self.set_params(self.latest_params)
            if list_form:
                loss, acc = [],[]
                length = len(self.val_data['x']) 
                for i in range(length):
                    x_i = self.val_data['x'][i].reshape(1,-1,)
                    y_i = self.val_data['y'][i].reshape(-1,)
                    loss_i, acc_i = self.model.evaluate(x_i,y_i, verbose=0)
                    loss.append(loss_i)
                    acc.append(acc_i)
            else:
                X, y_true = self.val_data['x'], self.val_data['y']
                loss, acc = self.model.evaluate(X, y_true, verbose=0)
            self.set_params(backup_params)
            return self.val_data['y'].shape[0], acc, loss
        else:
            return 0, 0, 0

    def has_uplink(self):
        if len(self.uplink) > 0:
            return True
        return False

    def has_downlink(self):
        if len(self.downlink) > 0:
            return True
        return False

    def add_downlink(self, nodes):
        if isinstance(nodes, list):
            self.downlink = list(set(self.downlink + nodes))
        if isinstance(nodes, Actor):
            self.downlink = list(set(self.downlink + [nodes]))
        return

    def add_uplink(self, nodes):
        if isinstance(nodes, list):
            self.uplink = list(set(self.uplink + nodes))
        if isinstance(nodes, Actor):
            self.uplink = list(set(self.uplink + [nodes]))
        return
    
    def delete_downlink(self, nodes):
        if isinstance(nodes, list):
            self.downlink = [c for c in self.downlink if c not in nodes]
        if isinstance(nodes, Actor):
            self.downlink.remove(nodes)
        return

    def delete_uplink(self, nodes):
        if isinstance(nodes, list):
            self.uplink = [c for c in self.uplink - nodes if c not in nodes]
        if isinstance(nodes, Actor):
            self.uplink.remove(nodes)
        return

    def clear_uplink(self):
        self.uplink.clear()
        return

    def clear_downlink(self):
        self.downlink.clear()
        return

    def set_uplink(self, nodes):
        self.clear_uplink()
        self.add_uplink(nodes)
        return

    def check_selected_trainable(self, selected_nodes):
        nodes_trainable = False
        valid_nodes = []
        for node in selected_nodes:
            if node in self.downlink:
                if node.check_trainable() == True:
                    nodes_trainable = True
                    valid_nodes.append(node)
        return nodes_trainable, valid_nodes

    def check_selected_testable(self, selected_nodes):
        nodes_testable = False
        valid_nodes = []
        for node in selected_nodes:
            if node in self.downlink:
                if node.check_testable() == True:
                    nodes_testable = True
                    valid_nodes.append(node)
        return nodes_testable, valid_nodes

    def test(self):
        return

    def train(self):
        return

    def check_trainable():
        return
    def check_testable():
        return
