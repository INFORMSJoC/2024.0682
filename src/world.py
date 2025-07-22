import os
from os.path import join
from warnings import simplefilter

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = join(ROOT_PATH, 'data')
FILE_PATH = join(ROOT_PATH, 'saved_models')
RESULT_PATH = join(ROOT_PATH, 'results')


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['Ciao', 'Epinions', 'Philadelphia', 'Tucson']
dataset = 'Epinions'
assert dataset in all_dataset
prepro = '2filter'
delete_ratio = 0
#uniform | xavier_uniform_ | kaiming_uniform_ | normal
config['initial_method'] = 'normal'

config['dec_ui'] = 'ui'
config['layer'] = 2
config['social_layer'] = 2
config['pop_num'] = 20
config['degree_num'] = 20
config['prior'] = True
config['pop_fading'] = 1
config['ci_alpha'] = 0.2
config['k'] = 20
config['latent_dim_rec'] = 64

config['bpr_batch_size'] = 1024
config['test_u_batch_size'] = 100

config['droprate'] = 0.5
config['lr'] = 0.001
config['emb_l2rg'] = 0.0001


GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
# device = torch.device("cpu")
seed = 23
LOAD = False
PATH = './saved_models'

config['device'] = device

TRAIN_epochs = 200
PATIENCE = 10
REPEAT = 10
ng_num = 4
topks = [10, 20, 30, 50, 100]


# testMethod = 'tfo'

simplefilter(action="ignore", category=FutureWarning)
