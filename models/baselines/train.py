import os
from sklearn.utils import compute_class_weight
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import GPT2TokenizerFast
from constants import WANDB, BERT_MODEL_NAME, LABEL_COLUMNS, LEARNING_RATE, MAX_TOKEN_COUNT, N_EPOCHS, BATCH_SIZE, MODEL_CKPT_PATH, TEST_SIZE, TRAIN_SIZE, VAL_SIZE
from models.baselines.DataModule import DataModule
from models.baselines.Model_CNN import Model_CNN
from models.baselines.Model_LSTM import Model_LSTM
from models.baselines.Model_GRU import Model_GRU


def train_model(model_name='from_scratch_model'):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pl.seed_everything(42)

    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    vocab_size = tokenizer.vocab_size
    embedd_dim = 300

    # Initialize DataModule including tokenizing data, splitting to train/val/test and batches
    data_module = DataModule(
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_COUNT
    )
    data_module.prepare_data()
    data_module.setup()

    # Compute the class weights
    dataset_df = data_module.train_df
    df = dataset_df[LABEL_COLUMNS].idxmax(axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=LABEL_COLUMNS,
        y=df
    )

    # Intialize model
    model = Model_LSTM(
        n_classes=len(LABEL_COLUMNS),
        vocab_size=vocab_size,
        embedding_dim=embedd_dim,
        class_weights=class_weights
    )

    # Checkpointing that saves the best model (based on validation loss)
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_CKPT_PATH,
        filename=f"{BERT_MODEL_NAME}-{model_name}",
        save_top_k=1,
        verbose=True,
        monitor="loss/val",
        mode="min"
    )

    CONFIG = dict(
        train_split=TRAIN_SIZE,
        val_split=VAL_SIZE,
        test_size=TEST_SIZE,
        model_name=BERT_MODEL_NAME,
        pretrained=True,
        num_classes=len(LABEL_COLUMNS),
        lr=LEARNING_RATE,
        max_token_count=MAX_TOKEN_COUNT,
        num_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    if WANDB:
        wandb_logger = WandbLogger(project="hs-bert", config=CONFIG)
        wandb_logger.watch(model, log="all")

    # Learning rate monitor and eary stopping callback
    early_stopping_callback = EarlyStopping(monitor='loss/val', patience=5)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        logger=wandb_logger if WANDB else None,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30
    )

    if WANDB:
        trainer.callbacks.append(lr_monitor)

    # Model training
    trainer.fit(model, data_module)

    # Model evaluation
    trainer.test()

    # Save best model as wandb artifact
    if WANDB:
        run = wandb.init(project='hs-bert', job_type='producer')

        artifact = wandb.Artifact(
            f"{BERT_MODEL_NAME}-{model_name}", type='model')
        artifact.add_file(
            f"{MODEL_CKPT_PATH}{BERT_MODEL_NAME}-{model_name}.ckpt")

        run.log_artifact(artifact)
        run.join()
