
import random
import json
from pathlib import Path
import tensorflow as tf
import numpy as np
import re
from skimage.util import random_noise
from skimage.transform import rotate

def read_json(train_data_dir, test_data_dir, group = 1, swap_label = 0, central = False,
        var = 0, salt = 0):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        train_data: dictionary of train (numpy) data
        test_data: dictionary of test (numpy) data
    '''
    
    clients = []
    train_npdata = {}
    test_npdata = {}

    train_files = Path.iterdir(train_data_dir)
    train_files = [f for f in train_files if f.suffix == '.json']

    for file_path in train_files:
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        train_npdata.update(cdata['user_data'])

    test_files = Path.iterdir(test_data_dir)
    test_files = [f for f in test_files if f.suffix == '.json']

    for file_path in test_files:
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_npdata.update(cdata['user_data'])
    clients = list(sorted(train_npdata.keys()))
    
    newclients = list()
    newtrain_data = dict()
    newtest_data = dict()
    if group == 0:
        if central:
            id = 0
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            i = 1
            while i < len(clients):
                train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                i += 1
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({0: train_data_c})
            newtest_data.update({0: test_data_c})
            newclients.append(0)
            return newclients, newtrain_data, newtest_data
        
        for id in range(int(len(clients))):
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id: train_data_c})
            newtest_data.update({id: test_data_c})
            newclients.append(id)
        return  newclients, newtrain_data, newtest_data
    
    if group == 1:
        if central:
            id = 0
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            i = 1
            while i < len(clients):
                train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                i += 1
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id // 10: train_data_c})
            newtest_data.update({id // 10: test_data_c})
            newclients.append(id // 10)
            return newclients, newtrain_data, newtest_data
        group_id = 0
        for id in range(0,int(len(clients)), 10):
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            i = 1
            while i < 10:
                train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                i += 1
                
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id // 10: train_data_c})
            newtest_data.update({id // 10: test_data_c})
            newclients.append(id // 10)
        return  newclients, newtrain_data, newtest_data
    if group == 2:
        if central:
            id = 0
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            i = 2
            while i < len(clients):
                train_data_x = np.concatenate((train_data_x, train_npdata[clients[i]]['x']))
                train_data_y = np.concatenate((train_data_y, train_npdata[clients[i]]['y']))
                test_data_x = np.concatenate((test_data_x, test_npdata[clients[i]]['x']))
                test_data_y = np.concatenate((test_data_y, test_npdata[clients[i]]['y']))
                i += 2
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id: train_data_c})
            newtest_data.update({id: test_data_c})
            newclients.append(id)
            
            id = 1
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            i = 3
            while i < len(clients):
                train_data_x = np.concatenate((train_data_x, train_npdata[clients[i]]['x']))
                train_data_y = np.concatenate((train_data_y, train_npdata[clients[i]]['y']))
                test_data_x = np.concatenate((test_data_x, test_npdata[clients[i]]['x']))
                test_data_y = np.concatenate((test_data_y, test_npdata[clients[i]]['y']))
                i += 2
            
            for i in range(len(train_data_y)):
                train_data_y[i] = (train_data_y[i] + 1) % 10

            for i in range(len(test_data_y)):
                test_data_y[i] = (test_data_y[i] + 1) % 10
                    
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id: train_data_c})
            newtest_data.update({id: test_data_c})
            newclients.append(id)
            return newclients, newtrain_data, newtest_data
        group_id = 0
        for id in range(0,int(len(clients)),10):
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            i = 1
            while i < 10:
                train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                i += 1
            for i in range(len(train_data_y)):
                if (id // 10) % 2 == 0:
                    train_data_y[i] = (train_data_y[i] + 1) % 10

            for i in range(len(test_data_y)):
                if (id // 10) % 2 == 0:
                    test_data_y[i] = (test_data_y[i] + 1) % 10

            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id//10: train_data_c})
            newtest_data.update({id//10: test_data_c})
            newclients.append(id // 10)
            group_id += 1
        return  newclients, newtrain_data, newtest_data
    
    
    # Different scope of input data
    # even id clients have digit 0-4, odd has 5-9
     
    if group == 3:
        group_id = 0
        size = 10
        for id in range(0,int(len(clients)),size):
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            i = 1
            if (id // size) % 2 == 0:
                while i < 10:
                    train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                    train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                    test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                    test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                    i += 1
                train_other_idx = []
                for i in range(len(train_data_x)):
                    if train_data_y[i] > 4:
                        digit = random.randint(0,9)
                        train_data_y[i] = digit
                train_data_x = train_data_x
                train_data_y = train_data_y
                test_other_idx = []
                for i in range(len(test_data_x)):
                    if test_data_y[i] > 4:
                        test_other_idx.append(i)
                test_data_x = np.delete(test_data_x, test_other_idx, axis=0)
                test_data_y = np.delete(test_data_y, test_other_idx, axis=0)
            else:
                while i < size:
                    train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                    train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                    test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                    test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                    i += 1
                train_other_idx = []
                for i in range(len(train_data_x)):
                    if train_data_y[i] < 5:
                        digit = random.randint(0,9)
                        train_data_y[i] = digit
                train_data_x = train_data_x
                train_data_y = train_data_y
                test_other_idx = []
                for i in range(len(test_data_x)):
                    if test_data_y[i] < 5:
                        test_other_idx.append(i)
                test_data_x = np.delete(test_data_x, test_other_idx, axis=0)
                test_data_y = np.delete(test_data_y, test_other_idx, axis=0)

            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id//size: train_data_c})
            newtest_data.update({id//size: test_data_c})
            newclients.append(id // size)
            group_id += 1
        if central:
            ctrain_data = newtrain_data
            ctest_data = newtest_data
            cclients = newclients
            newtrain_data = dict()
            newtest_data = dict()
            newclients = []
            # Need to construct 2 clients
            # 1st client: all odd id client
            id = 0
            train_data_x = []
            train_data_y = []
            test_data_x = []
            test_data_y = []
            for i in range(1, len(cclients),2):
                if train_data_x == []:
                    train_data_x = ctrain_data[cclients[i]]['x']
                    train_data_y = ctrain_data[cclients[i]]['y']
                    test_data_x = ctest_data[cclients[i]]['x']
                    test_data_y = ctest_data[cclients[i]]['y']
                    continue
                train_data_x = np.concatenate((train_data_x, ctrain_data[cclients[i]]['x']))
                train_data_y = np.concatenate((train_data_y, ctrain_data[cclients[i]]['y']))
                test_data_x = np.concatenate((test_data_x, ctest_data[cclients[i]]['x']))
                test_data_y = np.concatenate((test_data_y, ctest_data[cclients[i]]['y']))
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id: train_data_c})
            newtest_data.update({id: test_data_c})
            newclients.append(id)

            # 2nd client: all even id client
            id = 1
            train_data_x = []
            train_data_y = []
            test_data_x = []
            test_data_y = []
            for i in range(0, len(cclients),2):
                if train_data_x == []:
                    train_data_x = ctrain_data[cclients[i]]['x']
                    train_data_y = ctrain_data[cclients[i]]['y']
                    test_data_x = ctest_data[cclients[i]]['x']
                    test_data_y = ctest_data[cclients[i]]['y']
                    continue
                train_data_x = np.concatenate((train_data_x, ctrain_data[cclients[i]]['x']))
                train_data_y = np.concatenate((train_data_y, ctrain_data[cclients[i]]['y']))
                test_data_x = np.concatenate((test_data_x, ctest_data[cclients[i]]['x']))
                test_data_y = np.concatenate((test_data_y, ctest_data[cclients[i]]['y']))
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id: train_data_c})
            newtest_data.update({id: test_data_c})
            newclients.append(id)


            return newclients, newtrain_data, newtest_data
        return  newclients, newtrain_data, newtest_data
    if group == 6:
        group_id = 0
        for id in range(0,int(len(clients)),10):
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            i = 1
            while i < 10:
                train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                i += 1
            for i in range(len(train_data_x)):
                if (id // 10) % 2 == 0:
                    train_data_x[i] = random_noise(train_data_x[i], mode='gaussian', mean=0, var=var, clip=True, seed = i)
                    #train_data_x[i] = random_noise(train_data_x[i], mode='speckle', mean=0, var=var_s, clip=True, seed = i)
                    train_data_x[i] = random_noise(train_data_x[i], mode='s&p', amount = salt, clip=True, seed = i)

            for i in range(len(test_data_y)):
                if (id // 10) % 2 == 0:
                    test_data_x[i] = random_noise(test_data_x[i], mode='gaussian', mean=0, var=var, clip=True, seed = i)
                    #test_data_x[i] = random_noise(test_data_x[i], mode='speckle', mean=0, var=var_s, clip=True, seed = i)
                    test_data_x[i] = random_noise(test_data_x[i], mode='s&p', amount = salt, clip=True, seed = i)
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id//10: train_data_c})
            newtest_data.update({id//10: test_data_c})
            newclients.append(id // 10)
            group_id += 1
        if central:
            ctrain_data = newtrain_data
            ctest_data = newtest_data
            cclients = newclients
            newtrain_data = dict()
            newtest_data = dict()
            newclients = []
            # Need to construct 4 clients
            # 1st client: all unchanged client, odd id client
            id = 0
            train_data_x = []
            train_data_y = []
            test_data_x = []
            test_data_y = []
            for i in range(1, len(cclients),2):
                if train_data_x == []:
                    train_data_x = ctrain_data[cclients[i]]['x']
                    train_data_y = ctrain_data[cclients[i]]['y']
                    test_data_x = ctest_data[cclients[i]]['x']
                    test_data_y = ctest_data[cclients[i]]['y']
                    continue
                train_data_x = np.concatenate((train_data_x, ctrain_data[cclients[i]]['x']))
                train_data_y = np.concatenate((train_data_y, ctrain_data[cclients[i]]['y']))
                test_data_x = np.concatenate((test_data_x, ctest_data[cclients[i]]['x']))
                test_data_y = np.concatenate((test_data_y, ctest_data[cclients[i]]['y']))
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id: train_data_c})
            newtest_data.update({id: test_data_c})
            newclients.append(id)
            #2nd client: all client. test data is changed client: even id
            id = 1
            train_data_x = []
            train_data_y = []
            test_data_x = []
            test_data_y = []
            for i in range(len(cclients)):
                if i == 0:
                    train_data_x = ctrain_data[cclients[i]]['x']
                    train_data_y = ctrain_data[cclients[i]]['y']
                    test_data_x = ctest_data[cclients[i]]['x']
                    test_data_y = ctest_data[cclients[i]]['y']
                    continue
                train_data_x = np.concatenate((train_data_x, ctrain_data[cclients[i]]['x']))
                train_data_y = np.concatenate((train_data_y, ctrain_data[cclients[i]]['y']))
                if i % 2 == 0:
                    test_data_x = np.concatenate((test_data_x, ctest_data[cclients[i]]['x']))
                    test_data_y = np.concatenate((test_data_y, ctest_data[cclients[i]]['y']))
                    
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id: train_data_c})
            newtest_data.update({id: test_data_c})
            newclients.append(id)
            return newclients, newtrain_data, newtest_data
        return  newclients, newtrain_data, newtest_data
    # With even client has only 0,1 digit; odd client has every digit
    if group == 7:
        group_id = 0
        size = 10
        for id in range(0,int(len(clients)),size):
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            i = 1
            if (id // size) % 2 == 0:
                while i < 10:
                    train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                    train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                    test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                    test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                    i += 1
            else:
                while i < size:
                    train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                    train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                    test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                    test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                    i += 1
                
            if (id // size) % 2 == 0:
                for i in range(len(train_data_x)):
                    num = random.randint(0,9)
                    if num > 2:
                        digit = random.randint(0,9)
                        train_data_y[i] = digit
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id//size: train_data_c})
            newtest_data.update({id//size: test_data_c})
            newclients.append(id // size)
            group_id += 1
        if central:
            ctrain_data = newtrain_data
            ctest_data = newtest_data
            cclients = newclients
            newtrain_data = dict()
            newtest_data = dict()
            newclients = []
            # Need to construct 4 clients
            # 1st client: all unchanged client, odd id client
            id = 0
            train_data_x = []
            train_data_y = []
            test_data_x = []
            test_data_y = []
            for i in range(1, len(cclients),2):
                if train_data_x == []:
                    train_data_x = ctrain_data[cclients[i]]['x']
                    train_data_y = ctrain_data[cclients[i]]['y']
                    test_data_x = ctest_data[cclients[i]]['x']
                    test_data_y = ctest_data[cclients[i]]['y']
                    continue
                train_data_x = np.concatenate((train_data_x, ctrain_data[cclients[i]]['x']))
                train_data_y = np.concatenate((train_data_y, ctrain_data[cclients[i]]['y']))
                test_data_x = np.concatenate((test_data_x, ctest_data[cclients[i]]['x']))
                test_data_y = np.concatenate((test_data_y, ctest_data[cclients[i]]['y']))
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id: train_data_c})
            newtest_data.update({id: test_data_c})
            newclients.append(id)
            #2nd client: all client. test data is changed client: even id
            id = 1
            train_data_x = []
            train_data_y = []
            test_data_x = []
            test_data_y = []
            for i in range(len(cclients)):
                if i == 0:
                    train_data_x = ctrain_data[cclients[i]]['x']
                    train_data_y = ctrain_data[cclients[i]]['y']
                    test_data_x = ctest_data[cclients[i]]['x']
                    test_data_y = ctest_data[cclients[i]]['y']
                    continue
                train_data_x = np.concatenate((train_data_x, ctrain_data[cclients[i]]['x']))
                train_data_y = np.concatenate((train_data_y, ctrain_data[cclients[i]]['y']))
                if i % 2 == 0:
                    test_data_x = np.concatenate((test_data_x, ctest_data[cclients[i]]['x']))
                    test_data_y = np.concatenate((test_data_y, ctest_data[cclients[i]]['y']))
                    
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id: train_data_c})
            newtest_data.update({id: test_data_c})
            newclients.append(id)
            return newclients, newtrain_data, newtest_data
        return  newclients, newtrain_data, newtest_data
    if group == 8:
        group_id = 0
        size = 20
        for id in range(0,int(len(clients)),size):
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            i = 1
            if (id // size) % 2 == 0:
                while i < 5:
                    if id + i >= len(clients):
                        break
                    train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                    train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                    test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                    test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                    i += 1
            else:
                while i < size:
                    if id + i >= len(clients):
                        break
                    train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                    train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                    test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                    test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                    i += 1
                
            if (id // size) % 2 == 0:
                for i in range(len(train_data_x)):
                    if train_data_y[i] == 1 or train_data_y[i] == 3:
                        train_data_y[i] = 2
                for i in range(len(test_data_x)):
                    if test_data_y[i] == 1 or test_data_y[i] == 3:
                        test_data_y[i] = 2
            
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id//size: train_data_c})
            newtest_data.update({id//size: test_data_c})
            newclients.append(id // size)
            group_id += 1
        if central:
            ctrain_data = newtrain_data
            ctest_data = newtest_data
            cclients = newclients
            newtrain_data = dict()
            newtest_data = dict()
            newclients = []
            # Need to construct 4 clients
            # 1st client: all unchanged client, odd id client
            id = 0
            train_data_x = []
            train_data_y = []
            test_data_x = []
            test_data_y = []
            for i in range(1, len(cclients),2):
                if train_data_x == []:
                    train_data_x = ctrain_data[cclients[i]]['x']
                    train_data_y = ctrain_data[cclients[i]]['y']
                    test_data_x = ctest_data[cclients[i]]['x']
                    test_data_y = ctest_data[cclients[i]]['y']
                    continue
                train_data_x = np.concatenate((train_data_x, ctrain_data[cclients[i]]['x']))
                train_data_y = np.concatenate((train_data_y, ctrain_data[cclients[i]]['y']))
                test_data_x = np.concatenate((test_data_x, ctest_data[cclients[i]]['x']))
                test_data_y = np.concatenate((test_data_y, ctest_data[cclients[i]]['y']))
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id: train_data_c})
            newtest_data.update({id: test_data_c})
            newclients.append(id)
         
            #2nd client: all client. test data is changed client: even id
            id = 1
            train_data_x = []
            train_data_y = []
            test_data_x = []
            test_data_y = []
            for i in range(len(cclients)):
                if i == 0:
                    train_data_x = ctrain_data[cclients[i]]['x']
                    train_data_y = ctrain_data[cclients[i]]['y']
                    test_data_x = ctest_data[cclients[i]]['x']
                    test_data_y = ctest_data[cclients[i]]['y']
                    continue
                if i == 1:
                    train_data_x = np.concatenate((train_data_x, ctrain_data[cclients[i]]['x']))
                    train_data_y = np.concatenate((train_data_y, ctrain_data[cclients[i]]['y']))
                if i % 2 == 0:
                    train_data_x = np.concatenate((train_data_x, ctrain_data[cclients[i]]['x']))
                    train_data_y = np.concatenate((train_data_y, ctrain_data[cclients[i]]['y']))
                    test_data_x = np.concatenate((test_data_x, ctest_data[cclients[i]]['x']))
                    test_data_y = np.concatenate((test_data_y, ctest_data[cclients[i]]['y']))
                    
            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id: train_data_c})
            newtest_data.update({id: test_data_c})
            newclients.append(id)
            return newclients, newtrain_data, newtest_data
        return  newclients, newtrain_data, newtest_data
    if group == 4:
        if central:
            for id in [0,1,2,3]:
                train_data_x = train_npdata[clients[id]]['x']
                train_data_y = train_npdata[clients[id]]['y']
                test_data_x = test_npdata[clients[id]]['x']
                test_data_y = test_npdata[clients[id]]['y']
                i = id + 4
                while i < len(clients):
                    train_data_x = np.concatenate((train_data_x, train_npdata[clients[i]]['x']))
                    train_data_y = np.concatenate((train_data_y, train_npdata[clients[i]]['y']))
                    test_data_x = np.concatenate((test_data_x, test_npdata[clients[i]]['x']))
                    test_data_y = np.concatenate((test_data_y, test_npdata[clients[i]]['y']))
                    i += 4
                train_data_c = {'x': train_data_x, 'y': train_data_y}
                test_data_c = {'x': test_data_x, 'y': test_data_y}
                newtrain_data.update({id: train_data_c})
                newtest_data.update({id: test_data_c})
                newclients.append(id)
            return newclients, newtrain_data, newtest_data
        group_id = 0
        for id in range(0,int(len(clients)),10):
            train_data_x = train_npdata[clients[id]]['x']
            train_data_y = train_npdata[clients[id]]['y']
            test_data_x = test_npdata[clients[id]]['x']
            test_data_y = test_npdata[clients[id]]['y']
            i = 1
            while i < 10:
                train_data_x = np.concatenate((train_data_x, train_npdata[clients[id + i]]['x']))
                train_data_y = np.concatenate((train_data_y, train_npdata[clients[id + i]]['y']))
                test_data_x = np.concatenate((test_data_x, test_npdata[clients[id + i]]['x']))
                test_data_y = np.concatenate((test_data_y, test_npdata[clients[id + i]]['y']))
                i += 1
            for i in range(len(train_data_y)):
                if group_id % 4 == 1:
                    train_data_y[i] = (train_data_y[i] + 1) % 10
                if group_id % 4 == 2:
                    train_data_y[i] = (train_data_y[i] + 2) % 10
                if group_id % 4 == 3:
                    train_data_y[i] = (train_data_y[i] + 3) % 10

            for i in range(len(test_data_y)):
                if group_id % 4 == 1:
                    test_data_y[i] = (test_data_y[i] + 1) % 10
                if group_id % 4 == 2:
                    test_data_y[i] = (test_data_y[i] + 2) % 10
                if group_id % 4 == 3:
                    test_data_y[i] = (test_data_y[i] + 3) % 10

            train_data_c = {'x': train_data_x, 'y': train_data_y}
            test_data_c = {'x': test_data_x, 'y': test_data_y}
            newtrain_data.update({id//10: train_data_c})
            newtest_data.update({id//10: test_data_c})
            newclients.append(id // 10)
            group_id += 1
        return  newclients, newtrain_data, newtest_data

    return clients, train_npdata, test_npdata

def text2embs(dataset_list, emb_file, max_words=20):

    with open(emb_file, 'r') as inf:
        embs = json.load(inf)
    id2word = embs['vocab']
    word2id = {v: k for k,v in enumerate(id2word)}
    word_emb = np.array(embs['emba'])

    def _line_to_embs(line, w2d, d2e, max_words):
        word_list = re.findall(r"[\w']+|[.,!?;]", line)
        pad = int(max_words - len(word_list))
        pad_index = len(w2d)
        if pad <= 0:
            # Clip to max length
            word_list = word_list[:max_words]
        embs = []
        for word in word_list:
            if word in w2d:
                embs.append(d2e[w2d[word]])
            else:
                embs.append(d2e[pad_index])
        if pad > 0:
            # Add padding to the front of emb
            embs = [d2e[pad_index]]*pad + embs
        return embs

    new_dataset_list = []
    for dataset in dataset_list:
        for c, data in dataset.items():
            embs_list, labels_list = [], []
            for post, label in zip(data['x'], data['y']):
                embs = _line_to_embs(post[4], word2id, word_emb, max_words)
                embs_list.append(embs)
                labels_list += [1 if label=='4' else 0]
            dataset[c]['x'] = embs_list
            dataset[c]['y'] = labels_list
        new_dataset_list.append(dataset)
    return new_dataset_list

def read_mnist(train_data_dir, test_data_dir, group = 1, swap_label = 0, central = False):
    return read_json(train_data_dir, test_data_dir, group=group, swap_label = swap_label, central = central,var = 0.4, salt = 0.7)


def read_femnist(train_data_dir, test_data_dir, group = 1, swap_label = 0, central = False):
    return read_json(train_data_dir, test_data_dir, group=group, swap_label = swap_label, central = central,var = 1, salt = 0.6)

def read_femnist62(train_data_dir, test_data_dir, group = 1, swap_label = 0, central = False):
    return read_json(train_data_dir, test_data_dir, group=group, swap_label = swap_label, central = central)

def read_shakespeare(train_data_dir, test_data_dir, group = 1, swap_label = 0, central = False):
    return read_json(train_data_dir, test_data_dir, group=group, swap_label = swap_label, central = central)


def read_sent140(train_data_dir, test_data_dir, group = 1, swap_label = 0, central = False):
    return read_json(train_data_dir, test_data_dir, group=group, swap_label = swap_label, central = central)

def read_fmnist(train_data_dir, test_data_dir, group = 1, swap_label = 0, central = False):
    return read_json(train_data_dir, test_data_dir, group=group, swap_label = swap_label, central = central,var = 0.9, salt = 0.7)

def read_federated_data(dsname, group = 1, swap_label = 0, central = False):
    clients = []
    train_data = {}
    test_data = {}
    train_size, test_size = 0, 0
    wspath = Path(__file__).parent.parent.absolute() 
    train_data_dir = Path.joinpath(wspath, 'data', dsname, 'data', 'train').absolute()
    test_data_dir = Path.joinpath(wspath, 'data', dsname, 'data', 'test').absolute()

    if dsname.startswith('mnist'):
        clients, train_data, test_data = read_mnist(train_data_dir, test_data_dir, group =group, swap_label = swap_label, central = central)
    if dsname == 'emnist':
        clients, train_data, test_data = read_femnist(train_data_dir, test_data_dir, group =group, swap_label = swap_label, central = central)
    if dsname == 'femnist62':
        clients, train_data, test_data = read_femnist62(train_data_dir, test_data_dir, group =group, swap_label = swap_label, central = central)
    if dsname == 'fmnist':
        clients, train_data, test_data = read_fmnist(train_data_dir, test_data_dir, group =group, swap_label = swap_label, central = central)
    if dsname == 'shakespeare':
        clients, train_data, test_data = read_shakespeare(train_data_dir, test_data_dir, group =group, swap_label = swap_label, central = central)
    
    if dsname == 'sent140':
        max_words = 25
        emb_file = Path.joinpath(wspath, 'data', dsname, 'embs.json').absolute()
        clients, train_data, test_data = read_sent140(train_data_dir, test_data_dir, group =group, swap_label = swap_label, central = central)
        embs = text2embs([train_data, test_data], emb_file, max_words)
        train_data, test_data = embs[0], embs[1]
    # Convert list to numpy array
    
    for c in train_data.keys():
        train_data[c]['x'] = np.array(train_data[c]['x'], dtype=np.float32)
        train_data[c]['y'] = np.array(train_data[c]['y'], dtype=np.uint8)
        train_size += train_data[c]['y'].shape[0]
    for c in test_data.keys():
        test_data[c]['x'] = np.array(test_data[c]['x'], dtype=np.float32)
        test_data[c]['y'] = np.array(test_data[c]['y'], dtype=np.uint8)
        test_size += test_data[c]['y'].shape[0]
        
    # Print summary
    if group == 1:
        print("Symmetric Natural partition")
    if group == 2:
        print("Symmetric Synthetic predictor partition")
    if group == 3:
        print("Symmetric Synthetic label partition")
    if group == 0:
        print("Asymmetric natural partition")
    if group == 6:
        print("Asymmetric Synthetic input partition")
    if group == 8:
        print("Asymmetric Synthetic data size partition")
    
    print(f'Dataset: {dsname}')
    print(f'The dataset size: {train_size + test_size}, train size: {train_size}, test size: {test_size}.')
    print(f'Number of client: {len(train_data)}.')
    
    return clients, train_data, test_data


