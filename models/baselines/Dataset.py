import pandas as pd
import torch
from torch.utils.data import Dataset
from constants import LABEL_COLUMNS, TEXT_COLUMN
from transformers import GPT2TokenizerFast


class Dataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: GPT2TokenizerFast,
        max_token_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row[TEXT_COLUMN]
        labels = data_row[LABEL_COLUMNS]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt',
        )

        return (encoding["input_ids"].flatten(), torch.LongTensor(labels))
