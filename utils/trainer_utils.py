import numpy as np

'''
    Define the config of trainer,
    The type of trainer contain:  ['fedgroup', 'fesem', 'ifca', 'FAACL', 'FedDrift', 'Centralize']
'''
class TrainConfig(object):
    def __init__(self, dataset, model, trainer, group = 1, swap_label = 0, seed = 2077):
        self.trainer_type = trainer
        self.results_path = f'results/{dataset}/'
        self.group = group
        self.swap_label = swap_label
        self.trainer_config = {
            'dataset': dataset,
            'model': model,
            'seed': seed,
            'num_rounds': 300,
            'clients_per_round': 20,
            'eval_every': 1,
            'eval_locally': False,
            'dynamic': False, 
            'swap_p': 0, 
            'shift_type': None 
        }

        self.client_config = {
            'local_epochs': 10,
            'learning_rate': 0.001,
            'batch_size': 10,
            'temperature': None
        }

        if trainer in ['fedgroup', 'fesem', 'ifca', 'FAACL', 'FedDrift', 'Centralize']:
            if trainer == 'fedgroup':
                self.trainer_config.update({
                    'num_group': 3,
                    'group_agg_lr': 0.0,
                    'eval_global_model': True,
                    'pretrain_scale': 20,
                    'measure': 'EDC', 
                    'RAC': False,
                    'RCC': False,
                    'dynamic': True,
                    'temp_metrics': 'l2',
                    'temp_func': 'step', 
                    'temp_agg': False,
                    'recluster_epoch': None
                })
                
            if trainer in ['fesem',  'ifca']:
                self.trainer_config.update({
                    'num_group': 3,
                    'group_agg_lr': 0.0,
                    'eval_global_model': True
                })

            self.group_config = {
                'consensus': False,
                'max_clients': 999,
                'allow_empty': True
            }
        
        if self.trainer_config['dataset'] == 'emnist':
            self.client_config.update({'learning_rate': 0.003})
            self.trainer_config.update({'num_group': 5})

        if self.trainer_config['dataset'].startswith('mnist'):
            self.client_config.update({'learning_rate': 0.01})
            self.trainer_config.update({'num_group': 3})

        if self.trainer_config['dataset'] == 'fmnist':
            self.client_config.update({'learning_rate': 0.005})
            self.trainer_config.update({'num_group': 5})
        
        if self.trainer_config['dataset'] == 'femnist62':
            self.client_config.update({'learning_rate': 0.005})
            self.trainer_config.update({'num_group': 5})
        
        if trainer in ['FAACL', 'FedDrift', 'Centralize']:
            self.trainer_config.update({
                'num_group': 1,
                'group_agg_lr': 0.0,
                'eval_global_model': True
            })

def process_grad(grads):
    '''
    Args:
        grads: grad 
    Return:
        a flattened grad in numpy (1-D array)
    '''

    client_grads = grads[0] # shape = (784, 10)
    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array
        # (784, 10) append (10,)

    return client_grads

def calculate_cosine_dissimilarity(w1, w2):
    flat_w1, flat_w2 = process_grad(w1), process_grad(w2)
    cosine = np.dot(flat_w1, flat_w2) / (np.linalg.norm(flat_w1) * np.linalg.norm(flat_w2))
    dissimilarity = (1.0 - cosine) / 2.0 # scale to [0, 1] then flip
    return dissimilarity