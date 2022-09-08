
import wandb
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from utils.utils import split_stratified_into_train_val_test
from constants import DATASET_DIR, LABEL_COLUMNS, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, WANDB

from models.bert_based.Dataset import Dataset


class DataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def prepare_data(self, artifact_name='hs-dataset-preprocessed:latest', file_name='dataset_preprocessed.csv'):
        # Download dataset
        dataset_dir = DATASET_DIR

        if WANDB:
            run = wandb.init(project='hs-bert', job_type='model-training')
            dataset_artifact = run.use_artifact(artifact_name)
            dataset_dir = dataset_artifact.download()

        dataset_df = pd.read_csv(
            dataset_dir + '/' + file_name, encoding='utf-8')

        dataset_df = dataset_df.dropna()
        self.df = dataset_df

    def setup(self, stage=None):
        # Split to train/val/test
        self.train_df, self.val_df, self.test_df = split_stratified_into_train_val_test(
            self.df, stratify_colname=LABEL_COLUMNS, frac_train=TRAIN_SIZE, frac_val=VAL_SIZE, frac_test=TEST_SIZE
        )

        print(len(self.train_df), len(self.val_df), len(self.test_df))

        self.train_dataset = Dataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = Dataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

        self.val_dataset = Dataset(
            self.val_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )
