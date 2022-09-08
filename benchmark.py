
import os
import torch
import pandas as pd
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from benchmark.Dataset import Dataset
from transformers import BertTokenizerFast as BertTokenizer, logging
from torch.utils.data import DataLoader

from models.bert_based.Model_BERT_CNN import Model_BERT_CNN

logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
pl.seed_everything(42)

LABEL_COLUMNS = ['disability', 'gender', 'non hate',
                 'origin',  'religion', 'sexual_orientation']


def load_model(model_name: str):
    '''Loading trained model from checkpoint'''
    model = None

    if model_name == 'v4_cnn_freeze_6':
        model = Model_BERT_CNN.load_from_checkpoint(
            'benchmark/models/v4_cnn_freeze_6.ckpt')

    '''
    Only one model checkpoint (v4_cnn_freeze_6.ckpt) is included in 
    project due to large size of checkpoints. 

    If you with to benchmark other models, you can train desired models 
    again and add them to the "/benchmark/models/" in similar fashion as model above
    '''

    model.eval()
    model.freeze()
    return model


def load_dataset(dataset_name: str):
    '''Loading benchmark dataset'''
    if dataset_name == 'ethos_bin':
        df = pd.read_csv('benchmark/data/ethos_bin.csv',
                         sep=';', encoding='utf-8')

        df = df.assign(isHate=np.where(df['isHate'] > 0.0000, 1, 0))

        return df


# Initializing tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = load_model('v4_cnn_freeze_6')

# Initializing dataset
df = load_dataset('ethos_bin')
dataset = Dataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, num_workers=8)

# Metrics
f1_metric = torchmetrics.F1()
accuracy = torchmetrics.Accuracy()


# Testing loop
for i, batch in enumerate(dataloader):
    input_ids, attention_mask, labels, text = batch.values()

    output = model(input_ids, attention_mask)
    preds = torch.argmax(output, dim=1)

    '''Transforming multi-label output to match binary labels from benchmark dataset'''
    if labels.nelement() == 0:
        labels = torch.tensor([0])

    # 2nd index is "non-hate"
    p = torch.tensor([0]) if preds.item() == 2 else torch.tensor([1])
    l = torch.tensor([0]) if labels.item() == 0 else torch.tensor([1])

    acc = accuracy.update(p, l)
    f1 = f1_metric.update(p, l)

total_acc = accuracy.compute()
total_f1 = f1_metric.compute()

print(total_acc, total_f1)
