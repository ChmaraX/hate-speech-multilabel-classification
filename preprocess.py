import pandas as pd
import re
import wandb
import emoji
from constants import DATASET_DIR, DATASET_DIR, LABEL_COLUMNS, TEXT_COLUMN, WANDB
from utils.utils import calculateTokenCount, split_stratified_into_train_val_test


def log_dataset():
    '''Logs raw dataset as wandb artifact'''
    run = wandb.init(project='hs-bert', job_type='dataset-creation')

    df = pd.read_csv(DATASET_DIR + '/unified_dataset.csv', encoding='utf-8')

    artifact = wandb.Artifact('hs-dataset', type='dataset',
                              description='Raw hate-speech dataset', metadata={"size": len(df)})

    artifact.add_file(DATASET_DIR + '/unified_dataset.csv',
                      name="hs-dataset.csv")
    run.log_artifact(artifact)


def log_dev_dataset():
    '''Logs dev (smaller with same properties) dataset as wandb artifact'''
    dataset_dir = DATASET_DIR

    if WANDB:
        run = wandb.init(project='hs-bert', job_type='dataset-creation')

        dataset_artifact = run.use_artifact(
            'hs-dataset-preprocessed:latest')
        dataset_dir = dataset_artifact.download()

    df = pd.read_csv(
        dataset_dir + '/dataset_preprocessed.csv', encoding='utf-8')
    df = df.dropna()

    # dev will be 20% of full one
    _, _, dev = split_stratified_into_train_val_test(
        df, stratify_colname=LABEL_COLUMNS, frac_train=0.7, frac_val=0.1, frac_test=0.2
    )

    dev.to_csv(dataset_dir + '/dev-dataset_preprocessed.csv', index=False)

    # Upload dev dataset
    if WANDB:
        artifact = wandb.Artifact('dev-hs-dataset-preprocessed', type='dataset',
                                  description='Dev hate-speech dataset', metadata={"size": len(dev)})

        artifact.add_file(
            dataset_dir + '/dev-dataset_preprocessed.csv', name="dev-hs-dataset.csv")
        run.log_artifact(artifact)


def remove_handles(text):
    '''Removes Twitter specific handles (@user, RT)'''
    pattern = "RT|@[\w]+"
    return re.sub(pattern, "", text)


def remove_urls(text):
    '''Removes URLs'''
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return " ".join(re.sub(pattern, "", text).split())


def preprocess_and_log():
    '''Preprocess raw dataset and logs as wandb artifact'''

    # Download dataset
    raw_dataset_dir = DATASET_DIR

    if WANDB:
        run = wandb.init(project='hs-bert', job_type='dataset-preprocess')
        raw_dataset_artifact = run.use_artifact('hs-dataset:latest')
        raw_dataset_dir = raw_dataset_artifact.download()

    raw_dataset_df = pd.read_csv(
        raw_dataset_dir + '/unified_dataset.csv', encoding='utf-8')

    # Leave only columns we need
    df = raw_dataset_df[['text', 'target']].copy()

    # Remove Twitter handles
    df['clean_text'] = df.text.apply(lambda x: remove_handles(x))

    # Remove URLs
    df.clean_text = df.clean_text.apply(lambda x: remove_urls(x))

    # Substitute emojis with textual form
    df.clean_text = df.clean_text.apply(lambda x: emoji.demojize(x))

    # We don't need lowercase because BERT tokenizer does lowercasing
    # based on type of model (uncased, cased)
    # Lowercase
    # df.clean_text = df.clean_text.apply(lambda x: x.lower())

    # Count rows in each target category
    df_group_by = df.groupby(['target'])['target'].count()

    # Drop nan created from cleaning
    df = df.dropna()

    # One Hot encode
    one_hot = pd.get_dummies(df.target)
    df = df.drop('target', axis=1)
    df = df.join(one_hot)

    df.to_csv(raw_dataset_dir +
              '/dataset_preprocessed.csv', index=False)

    # Add metadata to artifact
    metadata = {
        'size': len(df),
        'token_len_99p': calculateTokenCount(df, TEXT_COLUMN)
    }
    metadata.update(df_group_by)

    print(metadata)

    # Upload preprocessed dataset
    if WANDB:
        artifact = wandb.Artifact('hs-dataset-preprocessed', type='dataset',
                                  description='Preprocessed hate-speech dataset',
                                  metadata=metadata)

        artifact.add_file(raw_dataset_dir +
                          '/dataset_preprocessed.csv')
        run.log_artifact(artifact)


preprocess_and_log()
# log_dataset()
# log_dev_dataset()
