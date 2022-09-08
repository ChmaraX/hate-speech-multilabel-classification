import sys

# from models.bert_based.train import train_model
from models.baselines.train import train_model

model_name = sys.argv[1]

train_model(model_name)
