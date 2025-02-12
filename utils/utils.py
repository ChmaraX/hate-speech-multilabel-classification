import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast as BertTokenizer
from constants import BERT_MODEL_NAME

def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    # if stratify_colname not in df_input.columns:
    #     raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[stratify_colname] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

def calculateTokenCount(df, column = 'text'):
    token_counts = []
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    for _, row in df.iterrows():
        token_count = len(tokenizer.encode(
            row[column], 
            max_length=512, 
            truncation=True
        ))
        token_counts.append(token_count)

    p = np.percentile(token_counts, 99)
    print('99th percentile (token count):', p)
    
    return p