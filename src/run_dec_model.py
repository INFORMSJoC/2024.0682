import time
import os
import torch
import pandas as pd
import Procedure
import utils
import sampler
from pprint import pprint
import dataloader
from src.models import my_graph_models
import world
import json


dataset = dataloader.DecGraphDataset(world.dataset)
print('===========config================')
pprint(world.config)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')
utils.set_seed(world.seed)
config = world.config
print(">>SEED:", world.seed)


file = utils.getFileName('CISGNN')
weight_path = os.path.join(world.FILE_PATH, dataset.dataset_name)
if not os.path.exists(weight_path):
    os.makedirs(weight_path, exist_ok=True)
result_path = os.path.join(world.RESULT_PATH,  dataset.dataset_name)
if not os.path.exists(result_path):
    os.makedirs(result_path, exist_ok=True)
result_file = os.path.join(result_path, f'{file}.json')
weight_file = os.path.join(weight_path, f'{file}.pth.tar')
all_metrics_df = pd.DataFrame()
best_hr = 0
print(f'#########Starting Experiment:{file}##############')
# ==============================
# torch.autograd.set_detect_anomaly(True)


for repeat_num in range(world.REPEAT):
    Recmodel = my_graph_models.CISGNN(config, dataset)
    Recmodel = Recmodel.to(world.device)
    bpr = sampler.BPRLoss(Recmodel, config)
    best_perf = {'hr@50': 0, 'ndcg@50': 0, 'best_epoch': 0}
    print(f"********** Run {repeat_num + 1} starts. **********")
    for epoch in range(1, world.TRAIN_epochs + 1):
        start = time.time()
        loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch)
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}][BPR aver loss{loss:.6f}]')
        val_results = Procedure.Evaluate(dataset, Recmodel, epoch, False)
        print('\t \t  Validation hr{:.4f}, ndcg{:.4f},' 
              'niche_rate{:.4f}, novelty{:3f}'.format(val_results['hr@50'],
                                                   val_results['ndcg@50'],
                                                   val_results['niche_rate@50'],
                                                   val_results['novelty@50']))
        if val_results['hr@50'] + 0.0001 > best_perf['hr@50']:
            best_perf['hr@50'] = val_results['hr@50']
            best_perf['ndcg@50'] = val_results['ndcg@50']
            best_perf['best_epoch'] = epoch
            torch.save(Recmodel.state_dict(), weight_file)
            print('\t [Increased] model saved')
        if epoch - best_perf['best_epoch'] > world.PATIENCE:
            print("early stop at %d epoch" % epoch)
            break
    print("[TEST]")
    Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
    test_results = Procedure.Test(dataset, Recmodel, False, False)
    print(test_results)
    with open(result_file, 'w') as file:
        file.write( json.dumps(test_results, indent=4))
