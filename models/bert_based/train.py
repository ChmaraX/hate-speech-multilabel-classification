import os
from sklearn.utils import compute_class_weight
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import BertModel, BertTokenizerFast as BertTokenizer
from models.bert_based.DataModule import DataModule
from constants import BERT_MODEL_NAME, FREEZE_LAYERS_COUNT, LABEL_COLUMNS, LEARNING_RATE, MAX_TOKEN_COUNT, N_EPOCHS, BATCH_SIZE, MODEL_CKPT_PATH, TEST_SIZE, TRAIN_SIZE, VAL_SIZE, WANDB
from models.bert_based.Model_BERT_LSTM import Model_BERT_LSTM
from models.bert_based.Model_BERT_Linear_v1 import Model_BERT_Linear_v1
from models.bert_based.Model_BERT_Linear_v01 import Model_BERT_Linear_v01
from models.bert_based.Model_BERT_Linear_v0 import Model_BERT_Linear_v0
from models.bert_based.Model_BERT_GRU import Model_BERT_GRU
from models.bert_based.Model_BERT_CNN import Model_BERT_CNN


def train_model(model_name):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pl.seed_everything(42)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # Initialize DataModule including tokenizing data, splitting to train/val/test and batches
    data_module = DataModule(
        tokenizer,
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_COUNT
    )
    data_module.prepare_data()
    data_module.setup()

    # Compute class weights
    dataset_df = data_module.train_df
    df = dataset_df[LABEL_COLUMNS].idxmax(axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=LABEL_COLUMNS,
        y=df
    )

    # Initialize pretrained BERT model
    bert_pretrained = BertModel.from_pretrained(
        BERT_MODEL_NAME, return_dict=True)

    # Freezing of selected layers
    if FREEZE_LAYERS_COUNT:
        # We freeze here the embeddings of the model
        print("Freezing embeddings")
        for param in bert_pretrained.embeddings.parameters():
            param.requires_grad = False

        if FREEZE_LAYERS_COUNT != -1:
            # if freeze_layer_count == -1, we only freeze the embedding layer
            # otherwise we freeze the first `freeze_layer_count` encoder layers
            for layer in bert_pretrained.encoder.layer[:FREEZE_LAYERS_COUNT]:
                print(layer)
                for param in layer.parameters():
                    param.requires_grad = False

        # Freezing pooler layer
        print("Freezing pooler")
        for param in bert_pretrained.pooler.parameters():
            param.requires_grad = False

    # Initialize instance of our model (including pre-trained BERT model)
    model = Model_BERT_Linear_v0(
        bert=bert_pretrained,
        n_classes=len(LABEL_COLUMNS),
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
        class_weights=class_weights,
        freeze_layer_count=FREEZE_LAYERS_COUNT,
    )

    if WANDB:
        wandb.config.update(CONFIG)
        wandb_logger = WandbLogger(project="hs-bert")
        wandb_logger.watch(model, log="all")

    # Learning rate monitor and eary stopping callback
    early_stopping_callback = EarlyStopping(monitor='loss/val', patience=3)
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
