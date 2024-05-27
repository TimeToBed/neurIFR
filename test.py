import os
import yaml
import argparse
#from models import *
import torch
from neurIFR.experiment import BSEXperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from fdataset import test_loader,test_loader2
import datetime

parser = argparse.ArgumentParser(description='Generic runner for models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/baseline_proto.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


model = torch.load(config['model_params']['pretrain'])
experiment = BSEXperiment(model, params=config["exp_params"])

runner = Trainer(**config['trainer_params'])
 
print(f"======= time {datetime.datetime.now()}=======")
print(f"======= test {config['model_params']['name']}  scale {config['data_params']['test_scale']}=======")
data_loader = test_loader2(**config["data_params"]) if config["exp_params"]["is_MRR"] else test_loader(**config["data_params"])
for _ in range(4):
    runner.test(experiment.eval(), dataloaders=data_loader)